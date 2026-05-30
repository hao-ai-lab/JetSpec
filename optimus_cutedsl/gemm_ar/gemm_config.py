# Copyright (C) 2025, Fri Dao.
import itertools
from typing import Optional, List, Literal
from functools import partial
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 128
    pingpong: bool = True
    cluster_m: int = 1
    cluster_n: int = 1
    persistent: bool = True
    num_ar_warps: int = 3
    swap_ab: bool = False
    # raster_order: int = 1
    max_swizzle_size: int = 8


def get_all_configs(
    device_capacity: Literal[9, 10] = 9,
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
    # tune_raster_order=True,
) -> List[GemmConfig]:
    assert device_capacity in [9, ]
    if device_capacity == 9:
        # tile_n_vals = [128, 160, 256]
        tile_mn_coop_vals = [
            (128, 128),
            (128, 256), 
            (64, 64),
        ]
        tile_mn_vals = []
        if tune_coop:
            tile_mn_vals += [(m, n, False) for m, n in tile_mn_coop_vals]
        cluster = [(1, 1)]
        swap_ab_vals = [False]
        persistent_vals = [False, True]
        configs: List[GemmConfig] = []
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab, persistent in itertools.product(
            tile_mn_vals,
            cluster,
            swap_ab_vals,
            persistent_vals,
            # raster_swizzle,
        ):
            num_ar_warps_vals = [3] if persistent else [-1]
            for num_ar_warps in num_ar_warps_vals:
                configs.append(
                    GemmConfig(
                        tile_m=tile_m,
                        tile_n=tile_n,
                        pingpong=pingpong,
                        cluster_m=cluster_m,
                        cluster_n=cluster_n,
                        swap_ab=swap_ab,
                        persistent=persistent,
                        num_ar_warps=num_ar_warps,
                        # raster_order=raster_order,
                        # max_swizzle_size=max_swizzle_size,
                    )
                )
        return configs
    # elif device_capacity == 10:
    #     tile_n_vals = [128, 160, 192, 224, 256]
    #     tile_n_64_vals = [128, 192, 256]
    #     tile_mn_cluster_vals = (
    #         [(128, tile_n, (1, 2)) for tile_n in tile_n_vals]
    #         # + [(128, tile_n, (2, 1)) for tile_n in tile_n_64_vals]
    #         + [(128, tile_n, (2, 1)) for tile_n in tile_n_vals]
    #         + [(256, tile_n, (2, 1)) for tile_n in tile_n_vals]
    #     )
    #     swap_ab_vals = [False, True]
    #     if epilogue in ["lse", "gated"]:
    #         swap_ab_vals = [False]
    #     max_swizzle_size_vals = [4, 8, 16]
    #     GemmConfigCls = partial(GemmConfig, pingpong=False)  # There's no pingpong on Sm100
    #     return [
    #         GemmConfigCls(
    #             tile_m=m, tile_n=n, cluster_m=cm, cluster_n=cn, swap_ab=sab, max_swizzle_size=ms
    #         )
    #         for (m, n, (cm, cn)), sab, ms in itertools.product(
    #             tile_mn_cluster_vals, swap_ab_vals, max_swizzle_size_vals
    #         )
    #     ]
