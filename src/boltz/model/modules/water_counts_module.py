import torch
from torch import nn
from torch.nn.functional import relu

import boltz.model.layers.initialize as init
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.modules.trunkv2 import (
    ContactConditioning,
)
from boltz.model.modules.utils import LinearNoBias


class WaterCountsModule(nn.Module):
    """Module for predicting per-residue water fractional credit.
    
    This module mirrors the structure of ConfidenceModule but is dedicated
    exclusively to water counts prediction, ensuring complete independence
    from confidence prediction logic.
    """

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        add_s_to_z_prod=False,
        add_s_input_to_s=False,
        add_z_input_to_z=False,
        maximum_bond_distance=0,
        bond_type_feature=False,
        water_counts_args: dict = None,
        compile_pairformer=False,
        fix_sym_check=False,
        cyclic_pos_enc=False,
        return_latent_feats=False,
        conditioning_cutoff_min=None,
        conditioning_cutoff_max=None,
        **kwargs,
    ):
        super().__init__()
        self.no_update_s = pairformer_args.get("no_update_s", False)

        self.s_to_z = LinearNoBias(token_s, token_z)
        self.s_to_z_transpose = LinearNoBias(token_s, token_z)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
            init.gating_init_(self.s_to_z_prod_out.weight)

        self.s_inputs_norm = nn.LayerNorm(token_s)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.add_s_input_to_s = add_s_input_to_s
        if add_s_input_to_s:
            self.s_input_to_s = LinearNoBias(token_s, token_s)
            init.gating_init_(self.s_input_to_s.weight)

        self.add_z_input_to_z = add_z_input_to_z
        if add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(
                token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
            )
            self.token_bonds = nn.Linear(
                1 if maximum_bond_distance == 0 else maximum_bond_distance + 2,
                token_z,
                bias=False,
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                from boltz.data import const
                self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z,
                cutoff_min=conditioning_cutoff_min,
                cutoff_max=conditioning_cutoff_max,
            )
        pairformer_args["v2"] = True
        self.pairformer_stack = PairformerModule(
            token_s,
            token_z,
            **pairformer_args,
        )
        self.return_latent_feats = return_latent_feats

        if water_counts_args is None:
            water_counts_args = {}
        self.water_counts_heads = WaterCountsHeads(
            token_s,
            token_z,
            **water_counts_args,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        feats,
        multiplicity=1,
        run_sequentially=False,
        use_kernels: bool = False,
    ):
        if run_sequentially and multiplicity > 1:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
            out_dicts = []
            for sample_idx in range(multiplicity):
                out_dicts.append(  # noqa: PERF401
                    self.forward(
                        s_inputs,
                        s,
                        z,
                        feats,
                        multiplicity=1,
                        run_sequentially=False,
                        use_kernels=use_kernels,
                    )
                )

            out_dict = {}
            for key in out_dicts[0]:
                out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
            return out_dict

        s_inputs = self.s_inputs_norm(s_inputs)
        if not self.no_update_s:
            s = self.s_norm(s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)

        z = self.z_norm(z)

        if self.add_z_input_to_z:
            relative_position_encoding = self.rel_pos(feats)
            z = z + relative_position_encoding
            z = z + self.token_bonds(feats["token_bonds"].float())
            if self.bond_type_feature:
                z = z + self.token_bonds_type(feats["type_bonds"].long())
            z = z + self.contact_conditioning(feats)

        s = s.repeat_interleave(multiplicity, 0)

        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        z = z.repeat_interleave(multiplicity, 0)
        s_inputs = s_inputs.repeat_interleave(multiplicity, 0)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s_t, z_t = self.pairformer_stack(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels
        )

        # AF3 has residual connections, we remove them
        s = s_t
        z = z_t

        out_dict = {}

        if self.return_latent_feats:
            out_dict["s_water_counts"] = s
            out_dict["z_water_counts"] = z

        # water counts heads
        out_dict.update(
            self.water_counts_heads(
                s=s,
                z=z,
                feats=feats,
                multiplicity=multiplicity,
            )
        )
        return out_dict


class WaterCountsHeads(nn.Module):
    """Minimal prediction head for water fractional credit.
    
    This head only outputs water_counts, in contrast to ConfidenceHeads
    which includes multiple prediction layers (PAE, PDE, pLDDT, resolved).
    """

    def __init__(
        self,
        token_s,
        token_z,
        **kwargs,
    ):
        super().__init__()
        # Token-level: output shape [B, N]
        self.to_water_counts = nn.Linear(token_s, 1, bias=True)
        # Initialize bias negative to start sparse (learnable threshold)
        nn.init.constant_(self.to_water_counts.bias, -3.0)

    def forward(
        self,
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        feats,
        multiplicity=1,
    ):
        water_counts_raw = self.to_water_counts(s)  # [B, N, 1]
        
        # Token-level: squeeze to [B, N]
        water_counts = water_counts_raw.squeeze(-1)
        
        # Apply ReLU to ensure water_counts are always positive
        water_counts = relu(water_counts)
        
        # Mask padded tokens to prevent contamination of loss
        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0).float()
        water_counts = water_counts * mask

        out_dict = {
            "water_counts": water_counts,
        }
        return out_dict
