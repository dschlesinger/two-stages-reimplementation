import { esmfold } from './esmfold.ts';
import { structureModule } from './structure.ts';
import type { Lane, ModelSpec, ModuleSpec, Source } from './types';

// Dimensions and operation order follow aqlaboratory/openfold (openfold/model/*.py,
// monomer config in openfold/config.py).

const r = String.raw;

// Reference implementation, pinned to one commit so the line anchors stay valid.
const REPO = 'https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/model';
const src = (symbol: string, file: string, line: number): Source => ({
	symbol,
	repo: 'aqlaboratory/openfold',
	file: `openfold/model/${file}`,
	url: `${REPO}/${file}#L${line}`
});

const M: Lane = { id: 'm', label: 'm  (S, L, 256)', kind: 'msa' };
const Z: Lane = { id: 'z', label: 'z  (L, L, 128)', kind: 'pair' };

const COL = { m: 0, mid: 270, z: 540, far: 810 };
const BLOCK = { x: -30, y: 520, width: 830, height: 1160 };
const IN_BLOCK = { m: COL.m - BLOCK.x, mid: COL.mid - BLOCK.x, z: COL.z - BLOCK.x };
const AFTER = BLOCK.y + BLOCK.height + 50;
const T = 20;

// The triangular updates and the structure module are the same computation as in ESMFold,
// which took them from OpenFold. Reuse that content and only change placement and citation.
function reuse(id: string, overrides: Partial<ModuleSpec>): ModuleSpec {
	const base = esmfold.modules.find((m) => m.id === id)!;
	return { ...base, paper: undefined, ...overrides };
}

const UTILS = 'https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/utils';
const util = (symbol: string, file: string, line: number): Source => ({
	symbol,
	repo: 'aqlaboratory/openfold',
	file: `openfold/utils/${file}`,
	url: `${UTILS}/${file}#L${line}`
});

const SM = structureModule({
	cols: { s: COL.m, mid: COL.mid, z: COL.z },
	y: AFTER + 170,
	inputs: [
		{ from: 't_s', kind: 'seq' },
		{ from: 't_z_out', kind: 'pair' }
	],
	sources: {
		structure: [src('StructureModule', 'structure_module.py', 817), src('StructureModule.forward', 'structure_module.py', 1161)],
		init: [src('StructureModule._forward_monomer', 'structure_module.py', 937)],
		ipa: [src('InvariantPointAttention', 'structure_module.py', 209), src('InvariantPointAttention.forward', 'structure_module.py', 304)],
		transition: [src('StructureModuleTransition', 'structure_module.py', 791), src('StructureModuleTransitionLayer', 'structure_module.py', 766)],
		backbone: [src('BackboneUpdate', 'structure_module.py', 736), util('Rigid.compose_q_update_vec', 'rigid_utils.py', 1009)],
		angles: [src('AngleResnet', 'structure_module.py', 78), src('AngleResnetBlock', 'structure_module.py', 50)],
		atoms: [
			util('torsion_angles_to_frames', 'feats.py', 185),
			util('frames_and_literature_positions_to_atom14_pos', 'feats.py', 253)
		]
	}
});

export const openfold: ModelSpec = {
	name: 'OpenFold',
	description:
		'A PyTorch reimplementation of AlphaFold2. The trunk is the 48-block Evoformer, which keeps an MSA state m (one row per aligned sequence) and a pairwise state z. The first row of m is the target sequence and becomes the single representation fed to the structure module.',
	notes:
		'Not drawn: the template embedder internals, the masked-MSA and experimentally-resolved heads, and recycling beyond the first pass.',
	initialView: ['t_msa', 't_tf', 't_residx', 'input_embedder', 'recycling', 'extra_msa', 't_m_in', 't_z_in'],

	modules: [
		{
			id: 'input_embedder',
			title: 'Input embedder',
			subtitle: 'MSA + target features + relative position',
			kind: 'cross',
			in: 'msa_feat (S, L, 49)   target_feat (L, 22)   residx (L,)',
			out: 'm (S, L, 256)   z (L, L, 128)',
			summary:
				'Builds both trunk states from the raw features. Every MSA row is embedded and the target sequence embedding is added to each row. The pairwise state starts as the sum of two embeddings of the target sequence, one for residue i and one for residue j, plus a relative position embedding.',
			position: { x: COL.mid, y: 80 },
			visual: {
				lanes: [M, Z],
				steps: [
					{ lane: 'm', label: 'Linear 49 -> 256 on every MSA row', shape: '(S, L, 256)' },
					{ lane: 'm', label: 'Linear 22 -> 256 on the target sequence', note: 'broadcast over all S rows', residual: true },
					{ lane: 'z', label: 'Linear 22 -> 128 for residue i  +  Linear 22 -> 128 for residue j', shape: '(L, L, 128)' },
					{ lane: 'z', label: 'One-hot of i - j clipped to [-32, 32], Linear 65 -> 128', residual: true }
				]
			},
			math: [
				{ label: 'MSA state', tex: r`m_{si} = W_{\mathrm{msa}}\, f^{\mathrm{msa}}_{si} + W_{\mathrm{tf}}\, f^{\mathrm{tf}}_{i}` },
				{
					label: 'Pair state',
					tex: r`z_{ij} = W_a f^{\mathrm{tf}}_{i} + W_b f^{\mathrm{tf}}_{j} + W_p\, \operatorname{onehot}\big(\operatorname{clip}(r_i - r_j, -32, 32)\big)`
				}
			],
			sources: [src('InputEmbedder', 'embedders.py', 39), src('InputEmbedder.forward', 'embedders.py', 108)],
			code: `class InputEmbedder(nn.Module):
    def __init__(self, tf_dim=22, msa_dim=49, c_z=128, c_m=256, relpos_k=32):
        super().__init__()
        self.relpos_k = relpos_k
        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_dim, c_m)
        self.linear_relpos = nn.Linear(2 * relpos_k + 1, c_z)

    def relpos(self, ri):                                            # (B, L) residue index
        d = ri[:, :, None] - ri[:, None, :]                          # (B, L, L), entry [b, i, j] = i - j
        d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k   # 0 .. 64
        return self.linear_relpos(F.one_hot(d.long(), 2 * self.relpos_k + 1).float())

    def forward(self, tf, ri, msa):                                  # (B, L, 22), (B, L), (B, S, L, 49)
        z = self.relpos(ri)                                          # (B, L, L, 128)
        z = z + self.linear_tf_z_i(tf)[:, :, None, :]                # varies with i
        z = z + self.linear_tf_z_j(tf)[:, None, :, :]                # varies with j

        m = self.linear_msa_m(msa)                                   # (B, S, L, 256)
        m = m + self.linear_tf_m(tf)[:, None, :, :]                  # same target embedding on every row
        return m, z`
		},
		{
			id: 'recycling',
			title: 'Recycling embedder',
			subtitle: 'Previous m row, z and distances',
			kind: 'cross',
			in: 'previous m[0] (L, 256), z (L, L, 128), positions (L, 3)',
			out: 'updates to m[0] and z',
			summary:
				'Feeds the previous pass back in: the normalised first MSA row is added to the first row of m, and the normalised pairwise state plus an embedding of the binned previous distances is added to z. On the first pass everything recycled is zero, so this only adds learned constants.',
			paper: 'The paper runs a single pass with recycling disabled.',
			position: { x: COL.mid, y: 250 },
			visual: {
				lanes: [M, Z],
				steps: [
					{ lane: 'm', label: 'LayerNorm(previous first row)', note: 'added to row 0 only', residual: true },
					{ lane: 'z', label: 'LayerNorm(previous z)', residual: true },
					{ lane: 'z', label: 'Previous C-beta distances into 15 bins, Linear 15 -> 128', residual: true }
				]
			},
			math: [
				{ label: 'First MSA row', tex: r`m_{1i} \leftarrow m_{1i} + \mathrm{LN}\big(m^{\mathrm{prev}}_{1i}\big)` },
				{
					label: 'Pair state',
					tex: r`z_{ij} \leftarrow z_{ij} + \mathrm{LN}\big(z^{\mathrm{prev}}_{ij}\big) + W_d\, \operatorname{onehot}\big(\operatorname{bin}(\lVert x^{\mathrm{prev}}_i - x^{\mathrm{prev}}_j \rVert)\big)`
				}
			],
			sources: [src('RecyclingEmbedder', 'embedders.py', 407), src('AlphaFold.iteration', 'model.py', 300)],
			code: `class RecyclingEmbedder(nn.Module):
    def __init__(self, c_m=256, c_z=128, min_bin=3.25, max_bin=20.75, no_bins=15):
        super().__init__()
        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear = nn.Linear(no_bins, c_z)
        self.register_buffer("bins", torch.linspace(min_bin, max_bin, no_bins))

    def forward(self, m1_prev, z_prev, x_prev):                      # (B, L, 256), (B, L, L, 128), (B, L, 3)
        lower = self.bins**2
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])])
        d2 = (x_prev[:, :, None] - x_prev[:, None, :]).pow(2).sum(-1, keepdim=True)   # (B, L, L, 1)
        one_hot = ((d2 > lower) & (d2 < upper)).float()              # (B, L, L, 15)

        m1_update = self.layer_norm_m(m1_prev)
        z_update = self.layer_norm_z(z_prev) + self.linear(one_hot)
        return m1_update, z_update


m1_update, z_update = recycling_embedder(m1_prev, z_prev, x_prev)    # all zeros on the first pass
m[:, 0] = m[:, 0] + m1_update                                        # only the target-sequence row
z = z + z_update`
		},
		{
			id: 'extra_msa',
			title: 'Templates and extra MSA',
			subtitle: 'Template embedder, extra MSA stack x 4',
			kind: 'pair',
			in: 'z (L, L, 128)   extra MSA (S_extra, L, 25)',
			out: 'z (L, L, 128)',
			summary:
				'Two optional sources of extra evidence that only write into z before the Evoformer. Template structures, when used, are embedded and added to z. The extra MSA stack runs four smaller Evoformer-like blocks over a large set of additional aligned sequences (width 64) and keeps only the pairwise state it produces.',
			position: { x: COL.z, y: 350 },
			visual: {
				lanes: [
					{ id: 'e', label: 'extra MSA  (S_extra, L, 64)', kind: 'msa' },
					Z
				],
				repeat: 'the four extra-MSA steps, x 4 blocks',
				steps: [
					{ lane: 'z', label: 'Template pair embedding (optional)', note: 'pointwise attention over templates', residual: true },
					{ lane: 'e', label: 'Linear 25 -> 64', shape: '(S_extra, L, 64)' },
					{ lane: 'all', label: 'Row attention with pair bias', note: '8 heads of width 8', residual: true },
					{ lane: 'e', label: 'Global column attention, transition', residual: true },
					{ lane: 'all', label: 'Outer product mean: extra MSA -> z', residual: true },
					{ lane: 'z', label: 'Triangular updates and pair transition', residual: true }
				]
			},
			math: [
				{ label: 'Templates', tex: r`z_{ij} \leftarrow z_{ij} + \mathrm{TemplatePointwiseAttn}\big(z_{ij}, \{t^{(k)}_{ij}\}_k\big)` },
				{ label: 'Extra MSA stack', tex: r`e = W_e f^{\mathrm{extra}}, \qquad (e, z) \leftarrow \mathrm{ExtraMSABlock}_{\ell}(e, z), \quad \ell = 1, \dots, 4` }
			],
			sources: [
				src('AlphaFold.iteration', 'model.py', 376),
				src('ExtraMSAEmbedder', 'embedders.py', 596),
				src('ExtraMSAStack', 'evoformer.py', 1067),
				src('TemplateEmbedder', 'embedders.py', 635)
			],
			code: `# Outline of the part of AlphaFold.iteration between recycling and the Evoformer.
if config.template.enabled:
    z = z + template_embedder(batch, z)                              # (B, L, L, 128)

if config.extra_msa.enabled:
    e = extra_msa_embedder(extra_msa_feat)                           # nn.Linear(25, 64) -> (B, S_extra, L, 64)
    z = extra_msa_stack(e, z)                                        # 4 blocks, only z is kept`
		},

		// ---- Evoformer block (repeated 48 times) ----
		{
			id: 'block',
			title: 'Evoformer block',
			subtitle: 'x 48, separate weights per block',
			kind: 'cross',
			in: 'm (S, L, 256)   z (L, L, 128)',
			out: 'm (S, L, 256)   z (L, L, 128)',
			summary:
				'One block of the Evoformer. The MSA state is updated first: row attention reads the pairwise state as a bias, column attention mixes information between aligned sequences. The updated MSA state is then written into the pairwise state by the outer product mean, and z is refined by the same triangular updates ESMFold uses.',
			position: { x: BLOCK.x, y: BLOCK.y },
			size: { width: BLOCK.width, height: BLOCK.height },
			visual: {
				lanes: [M, Z],
				repeat: 'x 48 blocks',
				steps: [
					{ lane: 'all', label: 'MSA row attention with pair bias', note: 'z -> bias on attention between residues', residual: true },
					{ lane: 'm', label: 'MSA column attention', note: 'between sequences at one position', residual: true },
					{ lane: 'm', label: 'MSA transition', note: '256 -> 1024 -> 256', residual: true },
					{ lane: 'all', label: 'Outer product mean: m -> z', shape: '(L, L, 128)', residual: true },
					{ lane: 'z', label: 'Triangle multiplication (outgoing)', residual: true },
					{ lane: 'z', label: 'Triangle multiplication (incoming)', residual: true },
					{ lane: 'z', label: 'Triangle attention (starting node)', residual: true },
					{ lane: 'z', label: 'Triangle attention (ending node)', residual: true },
					{ lane: 'z', label: 'Pair transition', note: '128 -> 512 -> 128', residual: true }
				]
			},
			math: [
				{
					label: 'MSA update',
					tex: r`\begin{aligned} m &\leftarrow m + \mathrm{RowAttn}(m,\ \mathrm{bias} = z) \\ m &\leftarrow m + \mathrm{ColAttn}(m) \\ m &\leftarrow m + \mathrm{Transition}_m(m) \end{aligned}`
				},
				{
					label: 'Pair update',
					tex: r`\begin{aligned} z &\leftarrow z + \mathrm{OuterProductMean}(m) \\ z &\leftarrow z + \mathrm{TriMul}_{\mathrm{out}}(z) \\ z &\leftarrow z + \mathrm{TriMul}_{\mathrm{in}}(z) \\ z &\leftarrow z + \mathrm{TriAttn}_{\mathrm{start}}(z) \\ z &\leftarrow z + \mathrm{TriAttn}_{\mathrm{end}}(z) \\ z &\leftarrow z + \mathrm{Transition}_z(z) \end{aligned}`
				}
			],
			sources: [src('EvoformerBlock', 'evoformer.py', 387), src('EvoformerBlock.forward', 'evoformer.py', 433), src('PairStack', 'evoformer.py', 126)],
			code: `class EvoformerBlock(nn.Module):
    def __init__(self, c_m=256, c_z=128):
        super().__init__()
        self.msa_att_row = MSARowAttentionWithPairBias(c_m, c_z, c_hidden=32, no_heads=8)
        self.msa_att_col = MSAColumnAttention(c_m, c_hidden=32, no_heads=8)
        self.msa_transition = Transition(c_m, n=4)
        self.outer_product_mean = OuterProductMean(c_m, c_z, c_hidden=32)
        # The five modules below sit under self.pair_stack in the source.
        self.tri_mul_out = TriangleMultiplicativeUpdate(c_z, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(c_z, outgoing=False)
        self.tri_att_start = TriangleAttention(c_z, c_hidden=32, no_heads=4, starting=True)
        self.tri_att_end = TriangleAttention(c_z, c_hidden=32, no_heads=4, starting=False)
        self.pair_transition = Transition(c_z, n=4)

    def forward(self, m, z):                                         # m: (B, S, L, 256), z: (B, L, L, 128)
        # MSA update. z is only read here, as an attention bias.
        m = m + self.msa_att_row(m, z)
        m = m + self.msa_att_col(m)
        m = m + self.msa_transition(m)

        # Pair update. It sees the m that was just updated above.
        z = z + self.outer_product_mean(m)
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_att_start(z)
        z = z + self.tri_att_end(z)
        z = z + self.pair_transition(z)
        return m, z`
		},
		{
			id: 'msa_att_row',
			title: 'MSA row attention',
			subtitle: '8 heads, pair-biased, gated',
			kind: 'cross',
			in: 'm (S, L, 256)   z (L, L, 128)',
			out: 'm update (S, L, 256)',
			summary:
				'Self-attention between residues, run separately inside every MSA row. The pairwise state is projected to one scalar per head and added to the attention logits, the same bias for every row. This is the only path from z into m, and plays the role pair2seq plays in ESMFold.',
			parent: 'block',
			position: { x: IN_BLOCK.m, y: 50 },
			visual: {
				lanes: [M, Z],
				steps: [
					{ lane: 'm', label: 'LayerNorm', shape: '(S, L, 256)' },
					{ lane: 'z', label: 'LayerNorm, Linear 128 -> 8, no bias', note: 'one bias map per head', shape: '(8, L, L)' },
					{ lane: 'm', label: 'q, k, v: Linear 256 -> 256, no bias', note: '8 heads of width 32' },
					{ lane: 'all', label: 'Logits = q . k / sqrt(32) + pair bias', note: 'attention over residues j, within each row s', shape: '(S, 8, L, L)' },
					{ lane: 'm', label: 'Gate sigmoid(Linear(m)), output Linear 256 -> 256', residual: true }
				]
			},
			math: [
				{ label: 'Pair bias', tex: r`b_{ij}^{h} = w_h^{\top}\,\mathrm{LN}(z_{ij}), \qquad h = 1, \dots, 8` },
				{ label: 'Attention within row s', tex: r`\alpha_{sij}^{h} = \operatorname{softmax}_j\!\left(\frac{q_{si}^{h} \cdot k_{sj}^{h}}{\sqrt{32}} + b_{ij}^{h}\right)` },
				{ label: 'Gated residual', tex: r`m_{si} \leftarrow m_{si} + W_o\Big( \sigma\big(W_g \hat m_{si} + b_g\big) \odot \Big[\textstyle\sum_j \alpha_{sij}^{h} v_{sj}^{h}\Big]_{h=1}^{8} \Big)` }
			],
			sources: [src('MSARowAttentionWithPairBias', 'msa.py', 319), src('MSAAttention', 'msa.py', 38), src('Attention', 'primitives.py', 360)],
			code: `class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c_m=256, c_z=128, c_hidden=32, no_heads=8):
        super().__init__()
        self.c_hidden, self.no_heads = c_hidden, no_heads
        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, no_heads, bias=False)         # pair bias
        # The five layers below sit under self.mha in the source.
        self.linear_q = nn.Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = nn.Linear(c_m, c_hidden * no_heads)          # gate
        self.linear_o = nn.Linear(c_hidden * no_heads, c_m)

    def split_heads(self, t):                                        # (B, S, L, 256) -> (B, S, 8, L, 32)
        return t.unflatten(-1, (self.no_heads, self.c_hidden)).transpose(-2, -3)

    def forward(self, m, z):                                         # m: (B, S, L, 256), z: (B, L, L, 128)
        x = self.layer_norm_m(m)
        bias = self.linear_z(self.layer_norm_z(z))                   # (B, L, L, 8)
        bias = bias.permute(0, 3, 1, 2).unsqueeze(1)                 # (B, 1, 8, L, L), shared by all rows

        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))

        logits = q @ k.transpose(-1, -2) / self.c_hidden**0.5 + bias # (B, S, 8, L, L)
        out = logits.softmax(dim=-1) @ v                             # (B, S, 8, L, 32)
        out = out.transpose(-2, -3).flatten(-2)                      # (B, S, L, 256)
        return self.linear_o(torch.sigmoid(self.linear_g(x)) * out)


# In the block:
m = m + msa_att_row(m, z)`
		},
		{
			id: 'msa_att_col',
			title: 'MSA column attention',
			subtitle: '8 heads, between sequences',
			kind: 'msa',
			in: 'm (S, L, 256)',
			out: 'm update (S, L, 256)',
			summary:
				'Self-attention between the aligned sequences at a single residue position. This is how evidence from homologous sequences reaches the first row, the target sequence. There is no pair bias here.',
			parent: 'block',
			position: { x: IN_BLOCK.m, y: 150 },
			visual: {
				lanes: [M],
				steps: [
					{ lane: 'm', label: 'Transpose (S, L) -> (L, S)', note: 'each column becomes a sequence of S items' },
					{ lane: 'm', label: 'LayerNorm, q / k / v', note: '8 heads of width 32' },
					{ lane: 'm', label: 'Attention over sequences t, at each position i', shape: '(L, 8, S, S)' },
					{ lane: 'm', label: 'Gate and output Linear 256 -> 256' },
					{ lane: 'm', label: 'Transpose back', residual: true }
				]
			},
			math: [
				{ label: 'Attention down column i', tex: r`\alpha_{sti}^{h} = \operatorname{softmax}_t\!\left(\frac{q_{si}^{h} \cdot k_{ti}^{h}}{\sqrt{32}}\right)` },
				{ label: 'Gated residual', tex: r`m_{si} \leftarrow m_{si} + W_o\Big( \sigma\big(W_g \hat m_{si} + b_g\big) \odot \Big[\textstyle\sum_t \alpha_{sti}^{h} v_{ti}^{h}\Big]_{h=1}^{8} \Big)` }
			],
			sources: [src('MSAColumnAttention', 'msa.py', 348), src('MSAAttention', 'msa.py', 38)],
			code: `class MSAColumnAttention(nn.Module):
    def __init__(self, c_m=256, c_hidden=32, no_heads=8):
        super().__init__()
        self.c_hidden, self.no_heads = c_hidden, no_heads
        # In the source these sit under self._msa_att (and its .mha).
        self.layer_norm_m = nn.LayerNorm(c_m)
        self.linear_q = nn.Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = nn.Linear(c_m, c_hidden * no_heads)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_m)

    def split_heads(self, t):                                        # (B, L, S, 256) -> (B, L, 8, S, 32)
        return t.unflatten(-1, (self.no_heads, self.c_hidden)).transpose(-2, -3)

    def forward(self, m):                                            # (B, S, L, 256)
        x = self.layer_norm_m(m.transpose(1, 2))                     # (B, L, S, 256): columns become rows

        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))

        logits = q @ k.transpose(-1, -2) / self.c_hidden**0.5        # (B, L, 8, S, S)
        out = logits.softmax(dim=-1) @ v
        out = out.transpose(-2, -3).flatten(-2)                      # (B, L, S, 256)
        out = self.linear_o(torch.sigmoid(self.linear_g(x)) * out)
        return out.transpose(1, 2)                                   # (B, S, L, 256)


# In the block:
m = m + msa_att_col(m)`
		},
		{
			id: 'msa_transition',
			title: 'MSA transition',
			subtitle: '256 -> 1024 -> 256',
			kind: 'msa',
			in: 'm (S, L, 256)',
			out: 'm update (S, L, 256)',
			summary: 'A feed-forward layer applied to every entry of the MSA state independently. Its output, added to m, is the MSA state that leaves the block.',
			parent: 'block',
			position: { x: IN_BLOCK.m, y: 250 },
			visual: {
				lanes: [M],
				steps: [
					{ lane: 'm', label: 'LayerNorm' },
					{ lane: 'm', label: 'Linear 256 -> 1024' },
					{ lane: 'm', label: 'ReLU' },
					{ lane: 'm', label: 'Linear 1024 -> 256', residual: true }
				]
			},
			math: [{ tex: r`m_{si} \leftarrow m_{si} + W_2\,\mathrm{ReLU}\big(W_1\,\mathrm{LN}(m_{si}) + b_1\big) + b_2` }],
			sources: [src('MSATransition', 'evoformer.py', 51)],
			code: `class Transition(nn.Module):
    """MSATransition and PairTransition in the source: the same layer at two widths."""

    def __init__(self, c, n=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c)
        self.linear_1 = nn.Linear(c, n * c)
        self.linear_2 = nn.Linear(n * c, c)

    def forward(self, x):
        return self.linear_2(F.relu(self.linear_1(self.layer_norm(x))))


msa_transition = Transition(c=256, n=4)
m = m + msa_transition(m)                                            # (B, S, L, 256)`
		},
		{
			id: 'outer_product_mean',
			title: 'Outer product mean',
			subtitle: 'MSA state -> pair state',
			kind: 'cross',
			in: 'm (S, L, 256)',
			out: 'z update (L, L, 128)',
			summary:
				'The only path from m to z. Every MSA entry is projected to two 32-wide vectors. For each residue pair, the outer product of the two residues\' vectors is averaged over all sequences, flattened to 1024 numbers, and projected to the pair width. It plays the role seq2pair plays in ESMFold, but pools over the whole alignment, so it can pick up co-variation between columns.',
			parent: 'block',
			position: { x: IN_BLOCK.mid, y: 420 },
			visual: {
				lanes: [M, Z],
				steps: [
					{ lane: 'm', label: 'LayerNorm' },
					{ lane: 'm', label: 'Two projections a and b: Linear 256 -> 32', shape: '2 x (S, L, 32)' },
					{ lane: 'all', label: 'Outer product a[s,i] x b[s,j], summed over sequences s', shape: '(L, L, 32, 32)' },
					{ lane: 'z', label: 'Flatten, Linear 1024 -> 128' },
					{ lane: 'z', label: 'Divide by the number of sequences', residual: true }
				]
			},
			math: [
				{ label: 'Projections', tex: r`a_{si} = W_a\,\mathrm{LN}(m_{si}) + b_a, \qquad b_{si} = W_b\,\mathrm{LN}(m_{si}) + b_b, \qquad a_{si}, b_{si} \in \mathbb{R}^{32}` },
				{ label: 'Outer product over the alignment', tex: r`o_{ij} = \operatorname{flatten}\Big( \sum_{s} a_{si} \otimes b_{sj} \Big) \in \mathbb{R}^{1024}` },
				{ label: 'Residual', tex: r`z_{ij} \leftarrow z_{ij} + \frac{W_o\, o_{ij} + b_o}{S + \epsilon}` }
			],
			sources: [src('OuterProductMean', 'outer_product_mean.py', 27), src('OuterProductMean.forward', 'outer_product_mean.py', 148)],
			code: `class OuterProductMean(nn.Module):
    def __init__(self, c_m=256, c_z=128, c_hidden=32, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden**2, c_z)

    def forward(self, m):                                            # (B, S, L, 256)
        m = self.layer_norm(m)
        a, b = self.linear_1(m), self.linear_2(m)                    # each (B, S, L, 32)
        outer = torch.einsum("bsic,bsjd->bijcd", a, b)               # sum over sequences: (B, L, L, 32, 32)
        outer = self.linear_out(outer.flatten(-2))                   # (B, L, L, 128)
        return outer / (m.shape[1] + self.eps)                       # mean over the S sequences


# In the block:
z = z + outer_product_mean(m)`
		},
		reuse('tri_mul_out', {
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 590 },
			sources: [src('TriangleMultiplicativeUpdate', 'triangular_multiplicative_update.py', 177), src('TriangleMultiplicationOutgoing', 'triangular_multiplicative_update.py', 561)]
		}),
		reuse('tri_mul_in', {
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 690 },
			sources: [src('TriangleMultiplicativeUpdate', 'triangular_multiplicative_update.py', 177), src('TriangleMultiplicationIncoming', 'triangular_multiplicative_update.py', 568)]
		}),
		reuse('tri_att_start', {
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 790 },
			sources: [src('TriangleAttention', 'triangular_attention.py', 32), src('Attention', 'primitives.py', 360)]
		}),
		reuse('tri_att_end', {
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 890 },
			sources: [src('TriangleAttention', 'triangular_attention.py', 32), src('PairStack.forward', 'evoformer.py', 180)]
		}),
		{
			id: 'pair_transition',
			title: 'Pair transition',
			subtitle: '128 -> 512 -> 128',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z update (L, L, 128)',
			summary: 'A feed-forward layer applied to every residue pair independently. Its output, added to z, is the pairwise state that leaves the block.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 990 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'LayerNorm' },
					{ lane: 'z', label: 'Linear 128 -> 512' },
					{ lane: 'z', label: 'ReLU' },
					{ lane: 'z', label: 'Linear 512 -> 128', residual: true }
				]
			},
			math: [{ tex: r`z_{ij} \leftarrow z_{ij} + W_2\,\mathrm{ReLU}\big(W_1\,\mathrm{LN}(z_{ij}) + b_1\big) + b_2` }],
			sources: [src('PairTransition', 'pair_transition.py', 24)],
			code: `class Transition(nn.Module):
    """MSATransition and PairTransition in the source: the same layer at two widths."""

    def __init__(self, c, n=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c)
        self.linear_1 = nn.Linear(c, n * c)
        self.linear_2 = nn.Linear(n * c, c)

    def forward(self, x):
        return self.linear_2(F.relu(self.linear_1(self.layer_norm(x))))


pair_transition = Transition(c=128, n=4)
z = z + pair_transition(z)                                           # (B, L, L, 128)`
		},

		// ---- After the trunk ----
		{
			id: 'single_proj',
			title: 'First row to single',
			subtitle: 'Linear 256 -> 384',
			kind: 'seq',
			in: 'm (S, L, 256)',
			out: 's (L, 384)',
			summary:
				'Takes the first row of the final MSA state, which is the target sequence, and projects it to the single representation. The other MSA rows are discarded at this point: whatever they contributed has to be in row 0 or in z by now.',
			position: { x: COL.m, y: AFTER },
			visual: {
				lanes: [M],
				steps: [
					{ lane: 'm', label: 'Select row 0 (the target sequence)', shape: '(L, 256)' },
					{ lane: 'm', label: 'Linear 256 -> 384', shape: '(L, 384)' }
				]
			},
			math: [{ tex: r`s_i = W\, m_{1i} + b` }],
			sources: [src('EvoformerStack.__init__', 'evoformer.py', 876), src('EvoformerStack.forward', 'evoformer.py', 1062)],
			code: `linear = nn.Linear(256, 384)                                         # EvoformerStack.linear

s = linear(m[:, 0])                                                  # (B, L, 384)`
		},
		{
			id: 'distogram',
			title: 'Distogram head',
			subtitle: 'Linear 128 -> 64, symmetrised',
			kind: 'head',
			in: 'z (L, L, 128)',
			out: 'logits (L, L, 64)',
			summary: 'Predicts a distribution over 64 distance bins for every residue pair directly from the final pairwise state. The logits are added to their transpose so the result is symmetric.',
			position: { x: COL.z, y: AFTER },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Linear 128 -> 64', shape: '(L, L, 64)' },
					{ lane: 'z', label: 'Add transpose', note: 'logits[i,j] + logits[j,i]' }
				]
			},
			math: [{ tex: r`\ell_{ij} = W z_{ij} + W z_{ji} + 2b` }],
			sources: [src('DistogramHead', 'heads.py', 127)],
			code: `linear = nn.Linear(128, 64)                                          # DistogramHead.linear

logits = linear(z)                                                   # (B, L, L, 64)
logits = logits + logits.transpose(1, 2)                             # symmetric in (i, j)`
		},
		{
			id: 'ptm',
			title: 'TM-score head',
			subtitle: 'Linear 128 -> 64, pTM checkpoints only',
			kind: 'head',
			in: 'z (L, L, 128)',
			out: 'pTM scalar   PAE (L, L)',
			summary:
				'Predicts a distribution over the aligned error between every pair of residues, from which the predicted TM-score and aligned error are computed. Only the "ptm" checkpoints have this head, including finetuning_ptm_2, the one this project downloads.',
			position: { x: COL.far, y: AFTER },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Linear 128 -> 64', note: '64 error bins up to 31 A', shape: '(L, L, 64)' },
					{ lane: 'z', label: 'Expected error per pair, expected TM-score term', shape: 'PAE (L, L), pTM scalar' }
				]
			},
			math: [
				{ label: 'Error distribution', tex: r`p_{ij} = \operatorname{softmax}\big(W z_{ij} + b\big) \in \mathbb{R}^{64}` },
				{ label: 'Predicted TM-score', tex: r`\mathrm{pTM} = \max_i \frac{1}{L} \sum_j \sum_b p_{ij,b}\, \frac{1}{1 + (e_b / d_0(L))^2}` }
			],
			sources: [src('TMScoreHead', 'heads.py', 170)],
			code: `linear = nn.Linear(128, 64)                                          # TMScoreHead.linear

tm_logits = linear(z)                                                # (B, L, L, 64)
# pTM and the aligned error are computed from these logits in openfold/utils/loss.py.`
		},
		{
			id: 'plddt',
			title: 'pLDDT head',
			subtitle: 'Per-residue confidence',
			kind: 'head',
			in: 's (L, 384), last iteration',
			out: 'pLDDT (L,)',
			summary:
				'Predicts how accurate each residue is expected to be, as a distribution over 50 lDDT bins for its C-alpha atom. It reads the single state from the last structure-module iteration.',
			position: { x: COL.m, y: SM.bottom },
			visual: {
				lanes: [{ id: 'x', label: 'structure module state', kind: 'struct' }],
				steps: [
					{ lane: 'x', label: 'LayerNorm', shape: '(L, 384)' },
					{ lane: 'x', label: 'Linear 384 -> 128, ReLU' },
					{ lane: 'x', label: 'Linear 128 -> 128, ReLU' },
					{ lane: 'x', label: 'Linear 128 -> 50', shape: '(L, 50)' },
					{ lane: 'x', label: 'Softmax, expected bin centre', shape: '(L,)' }
				]
			},
			math: [{ tex: r`\mathrm{pLDDT}_{i} = 100 \sum_{b=1}^{50} c_b \; \operatorname{softmax}\big(\ell_{i}\big)_b, \qquad c_b = \tfrac{b - 0.5}{50}` }],
			sources: [src('PerResidueLDDTCaPredictor', 'heads.py', 100), src('AuxiliaryHeads.forward', 'heads.py', 55)],
			code: `class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, c_in=384, c_hidden=128, no_bins=50):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)
        self.linear_1 = nn.Linear(c_in, c_hidden)
        self.linear_2 = nn.Linear(c_hidden, c_hidden)
        self.linear_3 = nn.Linear(c_hidden, no_bins)

    def forward(self, s):                                            # (B, L, 384), last structure-module state
        s = F.relu(self.linear_1(self.layer_norm(s)))
        s = F.relu(self.linear_2(s))
        return self.linear_3(s)                                      # (B, L, 50)


probs = lddt_head(single).softmax(dim=-1)
bin_centers = (torch.arange(50) + 0.5) / 50
plddt = 100 * (probs * bin_centers).sum(dim=-1)                      # (B, L)`
		},
		...SM.modules
	],

	tensors: [
		{
			id: 't_msa', label: 'MSA features', shape: '(S, L, 49)', kind: 'input', position: { x: COL.m + T, y: 0 },
			description:
				'The multiple sequence alignment: the target sequence in the first row and evolutionarily related sequences below it, aligned position by position. OpenFold needs this, unlike ESMFold. Each entry is a one-hot residue type plus deletion and cluster-profile features.',
			axes: [{ dim: 'S', meaning: 'aligned sequences (row 0 is the target)' }, { dim: 'L', meaning: 'residues' }, { dim: '49', meaning: 'residue one-hot, deletion features, cluster profile' }]
		},
		{
			id: 't_tf', label: 'Target features', shape: '(L, 22)', kind: 'input', position: { x: COL.mid + T, y: 0 },
			description: 'The target sequence on its own, as a one-hot vector per residue.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '22', meaning: 'one-hot residue type' }]
		},
		{
			id: 't_residx', label: 'Residue index', shape: '(L,) int', kind: 'input', position: { x: COL.z + T, y: 0 },
			description: 'The position of each residue along the chain. It is only used to work out how far apart two residues are in sequence.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: 'value', meaning: 'position in the chain' }]
		},
		{
			id: 't_m0', label: 'MSA state m', shape: '(S, L, 256)', kind: 'msa', position: { x: COL.m + T, y: 180 },
			description: 'The MSA state straight out of the input embedder: one vector per sequence per residue. Row 0 is the target sequence.',
			axes: [{ dim: 'S', meaning: 'aligned sequences' }, { dim: 'L', meaning: 'residues' }, { dim: '256', meaning: 'MSA width' }]
		},
		{
			id: 't_z0', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', position: { x: COL.z + T, y: 180 },
			description:
				'The pairwise state straight out of the input embedder. Unlike ESMFold, it already knows which amino acids residues i and j are, as well as their sequence separation.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_extra', label: 'Extra MSA features', shape: '(S_extra, L, 25)', kind: 'input', position: { x: COL.far + T, y: 250 },
			description: 'A much larger set of aligned sequences that did not fit in the main MSA, with a smaller feature set. They only influence the prediction through the pairwise state.',
			axes: [{ dim: 'S_extra', meaning: 'additional aligned sequences (up to a few thousand)' }, { dim: 'L', meaning: 'residues' }, { dim: '25', meaning: 'residue one-hot and deletion features' }]
		},
		{
			id: 't_m_in', label: 'MSA state m', shape: '(S, L, 256)', kind: 'msa', position: { x: COL.m + T, y: 450 },
			description: 'The MSA state entering a block. For the first block it is the embedder output plus the recycling term on row 0. For later blocks it is the previous block\'s output.',
			axes: [{ dim: 'S', meaning: 'aligned sequences' }, { dim: 'L', meaning: 'residues' }, { dim: '256', meaning: 'MSA width' }]
		},
		{
			id: 't_z_in', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', position: { x: COL.z + T, y: 450 },
			description: 'The pairwise state entering a block: one vector per ordered pair of residues.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_m_out', label: 'MSA state m', shape: '(S, L, 256)', kind: 'msa', parent: 'block', position: { x: IN_BLOCK.m + T, y: 350 },
			description: 'The MSA state leaving the block, after row attention, column attention and the transition. The outer product mean reads it in the same block.',
			axes: [{ dim: 'S', meaning: 'aligned sequences' }, { dim: 'L', meaning: 'residues' }, { dim: '256', meaning: 'MSA width' }]
		},
		{
			id: 't_dz', label: 'Pair update', shape: '(L, L, 128)', kind: 'cross', parent: 'block', position: { x: IN_BLOCK.mid + T, y: 520 },
			description: 'What the outer product mean writes into the pairwise state: a summary, pooled over all aligned sequences, of how the features at position i and position j go together.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_z_out', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', parent: 'block', position: { x: IN_BLOCK.z + T, y: 1090 },
			description: 'The pairwise state leaving the block. After the last block it goes to the structure module and the distogram and TM-score heads.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_s', label: 'Single state s', shape: '(L, 384)', kind: 'seq', position: { x: COL.m + T, y: AFTER + 100 },
			description: 'One vector per residue of the target sequence, taken from the first MSA row. This is OpenFold\'s per-residue representation, the counterpart of the sequence state in ESMFold.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '384', meaning: 'single width' }]
		},
		{
			id: 't_disto', label: 'Distogram logits', shape: '(L, L, 64)', kind: 'head', position: { x: COL.z + T, y: AFTER + 100 },
			description: 'A predicted distribution over the distance between every pair of residues, read directly from the final pairwise state.',
			axes: [{ dim: 'L, L', meaning: 'pair of residues' }, { dim: '64', meaning: 'distance bins' }]
		},
		{
			id: 't_pae', label: 'pTM, aligned error', shape: 'scalar, (L, L)', kind: 'head', position: { x: COL.far + T, y: AFTER + 100 },
			description: 'Confidence in the relative placement of residues. The aligned error is the expected position error of residue j when the prediction is aligned on residue i.',
			axes: [{ dim: 'scalar', meaning: 'pTM, between 0 and 1' }, { dim: 'L, L', meaning: 'predicted aligned error in angstroms' }]
		},
		{
			id: 't_plddt', label: 'pLDDT', shape: '(L,)', kind: 'head', position: { x: COL.m + T, y: SM.bottom + 100 },
			description: 'Per-residue confidence in the predicted structure, 0 to 100. High values mean the local environment of that residue is expected to be correct.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: 'value', meaning: 'expected lDDT of the C-alpha atom' }]
		},
		...SM.tensors
	],

	edges: [
		{ from: 't_msa', to: 'input_embedder', kind: 'input' },
		{ from: 't_tf', to: 'input_embedder', kind: 'input' },
		{ from: 't_residx', to: 'input_embedder', kind: 'input' },
		{ from: 'input_embedder', to: 't_m0', kind: 'msa' },
		{ from: 'input_embedder', to: 't_z0', kind: 'pair' },
		{ from: 't_m0', to: 'recycling', kind: 'msa' },
		{ from: 't_z0', to: 'recycling', kind: 'pair' },
		{ from: 'recycling', to: 't_m_in', kind: 'msa' },
		{ from: 'recycling', to: 'extra_msa', label: 'z', kind: 'pair' },
		{ from: 't_extra', to: 'extra_msa', kind: 'input' },
		{ from: 'extra_msa', to: 't_z_in', kind: 'pair' },

		{ from: 't_m_in', to: 'msa_att_row', kind: 'msa' },
		{ from: 't_z_in', to: 'msa_att_row', label: 'pair bias', kind: 'cross' },
		{ from: 'msa_att_row', to: 'msa_att_col', kind: 'msa' },
		{ from: 'msa_att_col', to: 'msa_transition', kind: 'msa' },
		{ from: 'msa_transition', to: 't_m_out', kind: 'msa' },
		{ from: 't_m_out', to: 'outer_product_mean', kind: 'msa' },
		{ from: 'outer_product_mean', to: 't_dz', kind: 'cross' },
		{ from: 't_dz', to: 'tri_mul_out', label: 'added to z', kind: 'cross' },
		{ from: 't_z_in', to: 'tri_mul_out', kind: 'pair' },
		{ from: 'tri_mul_out', to: 'tri_mul_in', kind: 'pair' },
		{ from: 'tri_mul_in', to: 'tri_att_start', kind: 'pair' },
		{ from: 'tri_att_start', to: 'tri_att_end', kind: 'pair' },
		{ from: 'tri_att_end', to: 'pair_transition', kind: 'pair' },
		{ from: 'pair_transition', to: 't_z_out', kind: 'pair' },

		{ from: 't_m_out', to: 'single_proj', kind: 'msa' },
		{ from: 'single_proj', to: 't_s', kind: 'seq' },
		{ from: 't_z_out', to: 'distogram', kind: 'pair' },
		{ from: 't_z_out', to: 'ptm', kind: 'pair' },
		{ from: 'distogram', to: 't_disto', kind: 'head' },
		{ from: 'ptm', to: 't_pae', kind: 'head' },
		{ from: 't_sm_s_out', to: 'plddt', kind: 'seq' },
		{ from: 'plddt', to: 't_plddt', kind: 'head' },
		...SM.edges
	]
};
