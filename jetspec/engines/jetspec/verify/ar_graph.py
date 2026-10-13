"""Single-token graph with device-side RoPE and paged-slot addressing."""
import torch

from jetspec.engines.jetspec.verify.graph_capture import GraphedVerify


class GraphedAR(GraphedVerify):
    """Capture the AR forward over a whole-prompt reservation.

    Each replay copies one token and advances one device position. Block tables
    are staged once per prompt; RoPE, node addresses and key length are computed
    inside the graph, using the same single-position math as the eager AR path.
    """

    def __init__(self, stack, cache, hidden_size, device, dtype):
        nlayers = len(stack.layers)
        super().__init__(
            stack, [cache.pool(i)[0] for i in range(nlayers)],
            [cache.pool(i)[1] for i in range(nlayers)],
            cache.reserved_block_table_width, stack.head_dim,
            hidden_size, device, dtype, buckets=(1,))
        self.g_position = torch.zeros((1,), dtype=torch.long, device=device)
        self.g_dummy = torch.zeros((1, 1, hidden_size), dtype=dtype, device=device)
        self.g_cu.copy_(torch.tensor([0, 1], dtype=torch.int32, device=device))
        self.g_qq_bias.zero_()

    def prepare(self, block_tables, prompt_len):
        """Bind this prompt's complete table without changing graph addresses."""
        torch.stack(block_tables, 0, out=self.g_block_tables)
        self.g_position.fill_(int(prompt_len))

    def _call_stack(self, B):
        cos, sin = self.stack.rotary_emb(self.g_dummy, self.g_position.view(1, 1))
        column = (self.g_position // self.stack.block_size).expand(self.nlayers, 1)
        node_blks = self.g_block_tables[:, 0].gather(1, column).to(torch.long)
        node_offs = (self.g_position % self.stack.block_size).expand(self.nlayers, 1)
        seq_lens_k = (self.g_position + 1).to(torch.int32)
        return self.stack(
            self.g_input_ids, cos, sin, self.k_pools, self.v_pools,
            [self.g_block_tables[i] for i in range(self.nlayers)],
            self.g_cu, seq_lens_k, self.g_qq_bias,
            [node_blks[i] for i in range(self.nlayers)],
            [node_offs[i] for i in range(self.nlayers)])

    @torch.inference_mode()
    def replay_ar(self, input_ids):
        self.g_input_ids.copy_(input_ids)
        if 1 not in self.graphs:
            self._capture_bucket(1)
        self.graphs[1].replay()
        # Outside capture so warmup/capture cannot advance the decode position.
        self.g_position.add_(1)
        return self.outputs[1]
