import { esmfold } from './esmfold.ts';
import type { Lane, ModelSpec, ModuleSpec, Source } from './types';

// Operation order follows jwohlwend/boltz (src/boltz/model/models/boltz1.py and the v1
// modules it imports). Widths are the Boltz-1 defaults: token_s 384, token_z 128, msa_s 64.

const r = String.raw;

// Reference implementation, pinned to one commit so the line anchors stay valid.
const REPO = 'https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/model';
const src = (symbol: string, file: string, line: number): Source => ({
	symbol,
	repo: 'jwohlwend/boltz',
	file: `src/boltz/model/${file}`,
	url: `${REPO}/${file}#L${line}`
});

const S: Lane = { id: 's', label: 's  (L, 384)', kind: 'seq' };
const Z: Lane = { id: 'z', label: 'z  (L, L, 128)', kind: 'pair' };

const COL = { s: 0, mid: 270, z: 540, far: 810 };
const BLOCK = { x: -30, y: 760, width: 830, height: 890 };
const IN_BLOCK = { s: COL.s - BLOCK.x, mid: COL.mid - BLOCK.x, z: COL.z - BLOCK.x };
const AFTER = BLOCK.y + BLOCK.height + 50;
const T = 20;
const DIFF = { x: -30, y: AFTER + 170, width: 830, height: 820 };
const IN_DIFF = { s: COL.s - DIFF.x, mid: COL.mid - DIFF.x, z: COL.z - DIFF.x };
const BELOW = DIFF.y + DIFF.height + 30;

const ATOMS: Lane = { id: 'x', label: 'atoms  (A, 128)', kind: 'struct' };
const TOKENS: Lane = { id: 'a', label: 'tokens a  (L, 768)', kind: 'seq' };
const COND: Lane = { id: 'c', label: 'conditioning', kind: 'cross' };

// Boltz's triangle attention is OpenFold's, the same computation as in ESMFold.
function reuse(id: string, overrides: Partial<ModuleSpec>): ModuleSpec {
	const base = esmfold.modules.find((m) => m.id === id)!;
	return { ...base, paper: undefined, ...overrides };
}

const TRANSITION = `class Transition(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))           # SwiGLU: one branch gates the other
`;

const TRI_MUL = `class TriangleMultiplication(nn.Module):
    """TriangleMultiplicationOutgoing and ...Incoming are two separate classes in the source."""

    def __init__(self, dim=128, outgoing=True):
        super().__init__()
        self.outgoing = outgoing
        self.norm_in = nn.LayerNorm(dim)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)              # both edges from one projection
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)              # and one gate
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def forward(self, z):                                            # (B, L, L, 128)
        x_in = self.norm_in(z)
        x = self.p_in(x_in) * self.g_in(x_in).sigmoid()              # (B, L, L, 256)
        a, b = x.chunk(2, dim=-1)                                    # each (B, L, L, 128)

        if self.outgoing:
            x = torch.einsum("bikd,bjkd->bijd", a, b)                # edges i->k and j->k
        else:
            x = torch.einsum("bkid,bkjd->bijd", a, b)                # edges k->i and k->j

        return self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
`;

const triMul = (outgoing: boolean): Pick<ModuleSpec, 'visual' | 'math' | 'code' | 'sources'> => ({
	visual: {
		lanes: [Z],
		steps: [
			{ lane: 'z', label: 'LayerNorm' },
			{ lane: 'z', label: 'Gated projection: Linear 128 -> 256 * sigmoid(Linear 128 -> 256), no bias', note: 'split in half into edges a and b', shape: '2 x (L, L, 128)' },
			{ lane: 'z', label: 'Sum over the third residue k', note: outgoing ? 'a[i,k] * b[j,k], per channel' : 'a[k,i] * b[k,j], per channel', shape: '(L, L, 128)' },
			{ lane: 'z', label: 'LayerNorm, Linear 128 -> 128, no bias' },
			{ lane: 'z', label: 'Output gate sigmoid(Linear(z))', residual: true }
		]
	},
	math: [
		{ label: 'Gated projection', tex: r`[a_{ij}, b_{ij}] = W_p \hat z_{ij} \odot \sigma\big(W_g \hat z_{ij}\big), \qquad \hat z = \mathrm{LN}(z)` },
		{
			label: outgoing ? 'Outgoing triangle product' : 'Incoming triangle product',
			tex: outgoing ? r`x_{ij} = \sum_{k} a_{ik} \odot b_{jk}` : r`x_{ij} = \sum_{k} a_{ki} \odot b_{kj}`
		},
		{ label: 'Gated residual', tex: r`z_{ij} \leftarrow z_{ij} + W_o\,\mathrm{LN}(x_{ij}) \odot \sigma\big(W_{g'} \hat z_{ij}\big)` }
	],
	sources: [
		outgoing
			? src('TriangleMultiplicationOutgoing', 'layers/triangular_mult.py', 39)
			: src('TriangleMultiplicationIncoming', 'layers/triangular_mult.py', 127)
	],
	code: `${TRI_MUL}

tri_mul = TriangleMultiplication(dim=128, outgoing=${outgoing ? 'True' : 'False'})
z = z + tri_mul(z)`
});

// Same computation as OpenFold's triangle attention, except that Boltz drops the bias on the gate
// and on the output projection.
const triAtt = (id: 'tri_att_start' | 'tri_att_end') =>
	esmfold.modules
		.find((m) => m.id === id)!
		.code.replace(
			'self.linear_g = nn.Linear(c_z, c_hidden * no_heads)          # gate',
			'self.linear_g = nn.Linear(c_z, c_hidden * no_heads, bias=False)   # gate'
		)
		.replace('self.linear_o = nn.Linear(c_hidden * no_heads, c_z)', 'self.linear_o = nn.Linear(c_hidden * no_heads, c_z, bias=False)');

const TRI_ATT_SOURCES = [
	src('TriangleAttention', 'layers/triangular_attention/attention.py', 33),
	src('Attention', 'layers/triangular_attention/primitives.py', 205)
];

export const boltz1: ModelSpec = {
	name: 'Boltz-1',
	description:
		'An open reimplementation of AlphaFold3. Inputs are tokens (one per residue for proteins). A small MSA module writes alignment information into the pairwise state, then the 48-block Pairformer refines a single state s and a pairwise state z. A diffusion model generates atom coordinates conditioned on both.',
	notes:
		'Not drawn: the internals of the input atom encoder and the confidence module, the windowing inside the atom transformers, and recycling beyond the first pass. Widths are the Boltz-1 defaults, which live in the checkpoint rather than the source.',
	initialView: ['t_atoms', 't_tok', 't_rel', 'input_embedder', 's_init', 'z_init', 'recycling', 't_s_in'],

	modules: [
		{
			id: 'input_embedder',
			title: 'Input embedder',
			subtitle: 'Atom encoder + token features',
			kind: 'input',
			in: 'atom features (A, ...)   token features (L, 71)',
			out: 's_inputs (L, 455)',
			summary:
				'Builds one fixed feature vector per token. A small attention network over the atoms of each token summarises its chemistry, and that summary is concatenated with the residue type, the MSA profile, the mean deletion count and a pocket flag. This vector is not updated by the trunk and is reused by the MSA module, the diffusion module and the confidence module.',
			position: { x: COL.mid, y: 80 },
			visual: {
				lanes: [
					{ id: 'a', label: 'atoms  (A, 128)', kind: 'input' },
					{ id: 't', label: 'tokens  (L, ...)', kind: 'seq' }
				],
				steps: [
					{ lane: 'a', label: 'Embed atom features and reference geometry', note: 'element, charge, reference position' },
					{ lane: 'a', label: 'Atom transformer', note: '3 blocks, attention in local windows of atoms' },
					{ lane: 'all', label: 'Pool atoms into their token', shape: '(L, 384)' },
					{ lane: 't', label: 'Concatenate residue type (33), profile (33), deletion mean (1), pocket (4)', shape: '(L, 455)' }
				]
			},
			math: [
				{ label: 'Atom summary', tex: r`a_i = \operatorname{mean}_{l \in \mathrm{atoms}(i)}\ \mathrm{ReLU}\big(W q_l\big), \qquad q = \mathrm{AtomTransformer}(\text{atom features})` },
				{ label: 'Token input', tex: r`s^{\mathrm{in}}_i = \big[\, a_i \,\Vert\, \mathrm{restype}_i \,\Vert\, \mathrm{profile}_i \,\Vert\, \mathrm{del}_i \,\Vert\, \mathrm{pocket}_i \,\big] \in \mathbb{R}^{455}` }
			],
			sources: [src('InputEmbedder', 'modules/trunk.py', 24), src('AtomAttentionEncoder', 'modules/encoders.py', 288)],
			code: `# Outline. AtomAttentionEncoder is a 3-block transformer over atoms; see the source for its internals.
def input_embedder(feats):
    a = atom_attention_encoder(feats)                                # (B, L, 384) one summary per token
    return torch.cat(
        [
            a,
            feats["res_type"],                                       # (B, L, 33) one-hot token type
            feats["profile"],                                        # (B, L, 33) MSA residue frequencies
            feats["deletion_mean"].unsqueeze(-1),                    # (B, L, 1)
            feats["pocket_feature"],                                 # (B, L, 4)
        ],
        dim=-1,
    )                                                                # (B, L, 455)`
		},
		{
			id: 's_init',
			title: 'Single initialisation',
			subtitle: 'Linear 455 -> 384, no bias',
			kind: 'seq',
			in: 's_inputs (L, 455)',
			out: 's_init (L, 384)',
			summary: 'A single linear projection of the token input features to the trunk width. This is the starting single state.',
			position: { x: COL.s, y: 250 },
			visual: {
				lanes: [S],
				steps: [{ lane: 's', label: 'Linear 455 -> 384, no bias', shape: '(L, 384)' }]
			},
			math: [{ tex: r`s^{\mathrm{init}}_i = W_s\, s^{\mathrm{in}}_i` }],
			sources: [src('Boltz1.__init__', 'models/boltz1.py', 157), src('Boltz1.forward', 'models/boltz1.py', 272)],
			code: `s_init_proj = nn.Linear(455, 384, bias=False)                        # Boltz1.s_init

s_init = s_init_proj(s_inputs)                                       # (B, L, 384)`
		},
		{
			id: 'z_init',
			title: 'Pair initialisation',
			subtitle: 'Token pair sum + position + bonds',
			kind: 'pair',
			in: 's_inputs (L, 455)   relative position, bonds',
			out: 'z_init (L, L, 128)',
			summary:
				'The starting pairwise state. Two linear projections of the token inputs are added, one for token i and one for token j, so z knows both residue identities from the start. A relative position encoding and an embedding of covalent bonds between tokens are added on top.',
			position: { x: COL.z, y: 250 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Linear 455 -> 128 for token i  +  Linear 455 -> 128 for token j', note: 'no bias', shape: '(L, L, 128)' },
					{ lane: 'z', label: 'Relative position encoding', note: 'residue offset, token offset, same-chain flag, chain offset: Linear 139 -> 128', residual: true },
					{ lane: 'z', label: 'Token bonds: Linear 1 -> 128', residual: true }
				]
			},
			math: [
				{ tex: r`z^{\mathrm{init}}_{ij} = W_1 s^{\mathrm{in}}_i + W_2 s^{\mathrm{in}}_j + W_p\, \mathrm{relpos}_{ij} + w_b\, \mathrm{bond}_{ij}` },
				{ label: 'Relative position features', tex: r`\mathrm{relpos}_{ij} = \big[\operatorname{onehot}_{66}(d^{\mathrm{res}}_{ij}) \,\Vert\, \operatorname{onehot}_{66}(d^{\mathrm{tok}}_{ij}) \,\Vert\, \mathbb{1}[\text{same entity}] \,\Vert\, \operatorname{onehot}_{6}(d^{\mathrm{chain}}_{ij})\big] \in \mathbb{R}^{139}` }
			],
			sources: [src('Boltz1.__init__', 'models/boltz1.py', 158), src('Boltz1.forward', 'models/boltz1.py', 272), src('RelativePositionEncoder', 'modules/encoders.py', 45)],
			code: `z_init_1 = nn.Linear(455, 128, bias=False)
z_init_2 = nn.Linear(455, 128, bias=False)
rel_pos = RelativePositionEncoder(token_z=128)                       # ends in nn.Linear(139, 128, bias=False)
token_bonds = nn.Linear(1, 128, bias=False)

z_init = z_init_1(s_inputs)[:, :, None] + z_init_2(s_inputs)[:, None, :]   # (B, L, L, 128)
z_init = z_init + rel_pos(feats)
z_init = z_init + token_bonds(feats["token_bonds"].float())          # token_bonds: (B, L, L, 1)`
		},
		{
			id: 'recycling',
			title: 'Recycling',
			subtitle: 'Initial state + projected previous state',
			kind: 'cross',
			in: 's_init, z_init, previous s and z',
			out: 's (L, 384)   z (L, L, 128)',
			summary:
				'Each pass starts from the initial states plus a normalised, linearly projected copy of the previous pass. On the first pass the previous states are zero and the projections have no bias, so only the LayerNorm bias passed through the projection is added.',
			paper: 'The paper runs a single pass with recycling disabled. Note that the Boltz command line defaults to 3 recycling steps.',
			position: { x: COL.mid, y: 420 },
			visual: {
				lanes: [S, Z],
				steps: [
					{ lane: 's', label: 's_init', shape: '(L, 384)' },
					{ lane: 'z', label: 'z_init', shape: '(L, L, 128)' },
					{ lane: 's', label: 'Linear(LayerNorm(previous s)), no bias', note: 'zeros on the first pass', residual: true },
					{ lane: 'z', label: 'Linear(LayerNorm(previous z)), no bias', note: 'zeros on the first pass', residual: true }
				]
			},
			math: [
				{ tex: r`s = s^{\mathrm{init}} + W_s\, \mathrm{LN}_s\big(s^{\mathrm{prev}}\big), \qquad z = z^{\mathrm{init}} + W_z\, \mathrm{LN}_z\big(z^{\mathrm{prev}}\big)` }
			],
			sources: [src('Boltz1.forward', 'models/boltz1.py', 272), src('Boltz1.__init__', 'models/boltz1.py', 178)],
			code: `s_norm, z_norm = nn.LayerNorm(384), nn.LayerNorm(128)
s_recycle = nn.Linear(384, 384, bias=False)
z_recycle = nn.Linear(128, 128, bias=False)

s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
for _ in range(recycling_steps + 1):                                 # the paper uses recycling_steps=0
    s = s_init + s_recycle(s_norm(s))
    z = z_init + z_recycle(z_norm(z))
    z = z + msa_module(z, s_inputs, feats)
    s, z = pairformer_module(s, z)                                   # 48 blocks`
		},
		{
			id: 'msa_module',
			title: 'MSA module',
			subtitle: 'x 4 blocks, writes into z',
			kind: 'msa',
			in: 'MSA features (S, L, 35)   s_inputs (L, 455)   z (L, L, 128)',
			out: 'z update (L, L, 128)',
			summary:
				'A short stack that reads the alignment and writes what it learns into the pairwise state. The MSA is embedded at width 64. In each block the MSA is updated by averaging over residues with weights that come from z, then the outer product mean writes it into z, and z gets one round of triangular updates. Only z is kept: the MSA state is thrown away after four blocks.',
			position: { x: COL.z, y: 590 },
			visual: {
				lanes: [{ id: 'm', label: 'm  (S, L, 64)', kind: 'msa' }, Z],
				repeat: 'x 4 blocks (after the two embedding steps)',
				steps: [
					{ lane: 'm', label: 'Embed MSA: Linear 35 -> 64, no bias', shape: '(S, L, 64)' },
					{ lane: 'm', label: 'Add Linear 455 -> 64 of s_inputs to every row', residual: true },
					{ lane: 'all', label: 'Pair-weighted averaging: z -> m', note: 'attention weights come only from z, 8 heads of width 32', residual: true },
					{ lane: 'm', label: 'MSA transition', note: '64 -> 256 -> 64, SwiGLU', residual: true },
					{ lane: 'all', label: 'Outer product mean: m -> z', note: 'hidden width 32', residual: true },
					{ lane: 'z', label: 'Triangle multiplication out, in', residual: true },
					{ lane: 'z', label: 'Triangle attention start, end', residual: true },
					{ lane: 'z', label: 'Pair transition', note: '128 -> 512 -> 128', residual: true }
				]
			},
			math: [
				{ label: 'Embedding', tex: r`m_{si} = W_m f^{\mathrm{msa}}_{si} + W_s s^{\mathrm{in}}_i` },
				{
					label: 'Pair-weighted averaging',
					tex: r`w_{ij}^{h} = \operatorname{softmax}_j\big(u_h^{\top}\,\mathrm{LN}(z_{ij})\big), \qquad m_{si} \leftarrow m_{si} + W_o\Big(\sigma(W_g \hat m_{si}) \odot \Big[\textstyle\sum_j w_{ij}^{h}\, W_v^{h} \hat m_{sj}\Big]_{h=1}^{8}\Big)`
				},
				{ label: 'Outer product mean', tex: r`z_{ij} \leftarrow z_{ij} + W_o\, \operatorname{flatten}\Big(\frac{1}{S}\sum_{s} a_{si} \otimes b_{sj}\Big) + b_o` }
			],
			sources: [
				src('MSAModule', 'modules/trunk.py', 116),
				src('MSALayer', 'modules/trunk.py', 292),
				src('PairWeightedAveraging', 'layers/pair_averaging.py', 7),
				src('OuterProductMean', 'layers/outer_product_mean.py', 7)
			],
			code: `class PairWeightedAveraging(nn.Module):
    def __init__(self, c_m=64, c_z=128, c_h=32, num_heads=8):
        super().__init__()
        self.c_h, self.num_heads = c_h, num_heads
        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)    # values
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)    # gate
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)          # attention logits, from z alone
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)

    def forward(self, m, z):                                         # m: (B, S, L, 64), z: (B, L, L, 128)
        m, z = self.norm_m(m), self.norm_z(z)
        v = self.proj_m(m).unflatten(-1, (self.num_heads, self.c_h)) # (B, S, L, 8, 32)
        w = self.proj_z(z).softmax(dim=2)                            # (B, L, L, 8), normalised over j
        o = torch.einsum("bijh,bsjhd->bsihd", w, v).flatten(-2)      # (B, S, L, 256)
        return self.proj_o(self.proj_g(m).sigmoid() * o)


class OuterProductMean(nn.Module):
    def __init__(self, c_in=64, c_hidden=32, c_out=128):
        super().__init__()
        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)

    def forward(self, m):                                            # (B, S, L, 64)
        m = self.norm(m)
        a, b = self.proj_a(m), self.proj_b(m)                        # each (B, S, L, 32)
        outer = torch.einsum("bsic,bsjd->bijcd", a, b).flatten(-2)   # (B, L, L, 1024)
        return self.proj_o(outer / m.shape[1])                       # mean over the S sequences


class MSALayer(nn.Module):
    def __init__(self, msa_s=64, token_z=128):
        super().__init__()
        self.pair_weighted_averaging = PairWeightedAveraging(msa_s, token_z, c_h=32, num_heads=8)
        self.msa_transition = Transition(msa_s, 4 * msa_s)
        self.outer_product_mean = OuterProductMean(msa_s, 32, token_z)
        self.tri_mul_out = TriangleMultiplication(token_z, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(token_z, outgoing=False)
        self.tri_att_start = TriangleAttention(token_z, 32, 4, starting=True)
        self.tri_att_end = TriangleAttention(token_z, 32, 4, starting=False)
        self.z_transition = Transition(token_z, 4 * token_z)

    def forward(self, z, m):
        m = m + self.pair_weighted_averaging(m, z)                   # z -> m
        m = m + self.msa_transition(m)
        z = z + self.outer_product_mean(m)                           # m -> z
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_att_start(z)
        z = z + self.tri_att_end(z)
        z = z + self.z_transition(z)
        return z, m


# MSAModule.forward, in outline:
m = msa_proj(msa_feat) + s_proj(s_inputs).unsqueeze(1)               # (B, S, L, 64)
for layer in layers:                                                 # 4 blocks
    z, m = layer(z, m)
# In the model: z = z + msa_module(z, s_inputs, feats)`
		},

		// ---- Pairformer block (repeated 48 times) ----
		{
			id: 'block',
			title: 'Pairformer block',
			subtitle: 'x 48, separate weights per block',
			kind: 'cross',
			in: 's (L, 384)   z (L, L, 128)',
			out: 's (L, 384)   z (L, L, 128)',
			summary:
				'One block of the Pairformer. The order is the reverse of ESMFold and OpenFold: the pairwise state is updated first, by triangular updates and a transition, and the single state is then updated by attention that uses the new z as a bias. There is no module that writes s into z inside the block. The single state only reaches z at initialisation and, indirectly, through the MSA module.',
			position: { x: BLOCK.x, y: BLOCK.y },
			size: { width: BLOCK.width, height: BLOCK.height },
			visual: {
				lanes: [S, Z],
				repeat: 'x 48 blocks',
				steps: [
					{ lane: 'z', label: 'Triangle multiplication (outgoing)', residual: true },
					{ lane: 'z', label: 'Triangle multiplication (incoming)', residual: true },
					{ lane: 'z', label: 'Triangle attention (starting node)', residual: true },
					{ lane: 'z', label: 'Triangle attention (ending node)', residual: true },
					{ lane: 'z', label: 'Pair transition', note: '128 -> 512 -> 128, SwiGLU', residual: true },
					{ lane: 'all', label: 'Attention with pair bias: z -> bias on s attention', note: '16 heads of width 24', residual: true },
					{ lane: 's', label: 'Single transition', note: '384 -> 1536 -> 384, SwiGLU', residual: true }
				]
			},
			math: [
				{
					label: 'Pair update',
					tex: r`\begin{aligned} z &\leftarrow z + \mathrm{TriMul}_{\mathrm{out}}(z) \\ z &\leftarrow z + \mathrm{TriMul}_{\mathrm{in}}(z) \\ z &\leftarrow z + \mathrm{TriAttn}_{\mathrm{start}}(z) \\ z &\leftarrow z + \mathrm{TriAttn}_{\mathrm{end}}(z) \\ z &\leftarrow z + \mathrm{Transition}_z(z) \end{aligned}`
				},
				{
					label: 'Single update',
					tex: r`\begin{aligned} s &\leftarrow s + \mathrm{AttnPairBias}(s,\ \mathrm{bias} = z) \\ s &\leftarrow s + \mathrm{Transition}_s(s) \end{aligned}`
				}
			],
			sources: [src('PairformerLayer', 'modules/trunk.py', 557), src('PairformerLayer.forward', 'modules/trunk.py', 613), src('PairformerModule', 'modules/trunk.py', 424)],
			code: `class PairformerLayer(nn.Module):
    def __init__(self, token_s=384, token_z=128, num_heads=16):
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(token_z, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(token_z, outgoing=False)
        self.tri_att_start = TriangleAttention(token_z, c_hidden=32, no_heads=4, starting=True)
        self.tri_att_end = TriangleAttention(token_z, c_hidden=32, no_heads=4, starting=False)
        self.transition_z = Transition(token_z, 4 * token_z)
        self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.transition_s = Transition(token_s, 4 * token_s)

    def forward(self, s, z):                                         # s: (B, L, 384), z: (B, L, L, 128)
        # Pair update first. Nothing here reads s.
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_att_start(z)
        z = z + self.tri_att_end(z)
        z = z + self.transition_z(z)

        # Single update. It sees the z that was just updated above, as an attention bias.
        s = s + self.attention(s, z)
        s = s + self.transition_s(s)
        return s, z`
		},
		{
			id: 'tri_mul_out',
			title: 'Triangle multiplication',
			subtitle: 'Outgoing edges',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z update (L, L, 128)',
			summary:
				'Updates the edge i-j from the two other edges of every triangle i-j-k, here the edges leaving i and j towards k. Boltz computes both edge projections with a single linear layer of twice the width and drops all biases.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 50 },
			...triMul(true)
		},
		{
			id: 'tri_mul_in',
			title: 'Triangle multiplication',
			subtitle: 'Incoming edges',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z update (L, L, 128)',
			summary: 'The mirror image of the outgoing update, with separate weights: edge i-j is updated from the edges arriving at i and j from every third residue k.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 150 },
			...triMul(false)
		},
		reuse('tri_att_start', { parent: 'block', position: { x: IN_BLOCK.z, y: 250 }, sources: TRI_ATT_SOURCES, code: triAtt('tri_att_start') }),
		reuse('tri_att_end', { parent: 'block', position: { x: IN_BLOCK.z, y: 350 }, sources: TRI_ATT_SOURCES, code: triAtt('tri_att_end') }),
		{
			id: 'transition_z',
			title: 'Pair transition',
			subtitle: '128 -> 512 -> 128, SwiGLU',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z update (L, L, 128)',
			summary: 'A gated feed-forward layer applied to every token pair independently. Its output, added to z, is the pairwise state that leaves the block.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 450 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'LayerNorm' },
					{ lane: 'z', label: 'Two projections: Linear 128 -> 512, no bias', note: 'SiLU on the first, multiply by the second' },
					{ lane: 'z', label: 'Linear 512 -> 128, no bias', residual: true }
				]
			},
			math: [{ tex: r`z_{ij} \leftarrow z_{ij} + W_3\Big(\mathrm{SiLU}\big(W_1 \hat z_{ij}\big) \odot W_2 \hat z_{ij}\Big), \qquad \hat z = \mathrm{LN}(z)` }],
			sources: [src('Transition', 'layers/transition.py', 8)],
			code: `${TRANSITION}

transition_z = Transition(dim=128, hidden=512)
z = z + transition_z(z)                                              # (B, L, L, 128)`
		},
		{
			id: 'attention',
			title: 'Attention with pair bias',
			subtitle: '16 heads, gated',
			kind: 'cross',
			in: 's (L, 384)   z (L, L, 128)',
			out: 's update (L, 384)',
			summary:
				'Self-attention between tokens of the single state. The pairwise state is projected to one scalar per head and added to the attention logits. This is the only path from z into s, the counterpart of pair2seq plus sequence attention in ESMFold.',
			parent: 'block',
			position: { x: IN_BLOCK.s, y: 620 },
			visual: {
				lanes: [S, Z],
				steps: [
					{ lane: 's', label: 'LayerNorm', shape: '(L, 384)' },
					{ lane: 'z', label: 'LayerNorm, Linear 128 -> 16, no bias', note: 'one bias map per head', shape: '(16, L, L)' },
					{ lane: 's', label: 'q: Linear 384 -> 384.  k, v: Linear 384 -> 384, no bias', note: '16 heads of width 24' },
					{ lane: 'all', label: 'Logits = q . k / sqrt(24) + pair bias', shape: '(16, L, L)' },
					{ lane: 's', label: 'Gate sigmoid(Linear(s)), output Linear 384 -> 384, no bias', residual: true }
				]
			},
			math: [
				{ label: 'Pair bias', tex: r`b_{ij}^{h} = w_h^{\top}\,\mathrm{LN}(z_{ij}), \qquad h = 1, \dots, 16` },
				{ label: 'Attention weights', tex: r`\alpha_{ij}^{h} = \operatorname{softmax}_j\!\left(\frac{q_i^{h} \cdot k_j^{h}}{\sqrt{24}} + b_{ij}^{h}\right)` },
				{ label: 'Gated residual', tex: r`s_i \leftarrow s_i + W_o\Big( \sigma\big(W_g \hat s_i\big) \odot \Big[\textstyle\sum_j \alpha_{ij}^{h} v_j^{h}\Big]_{h=1}^{16} \Big)` }
			],
			sources: [src('AttentionPairBias', 'layers/attention.py', 8), src('AttentionPairBias.forward', 'layers/attention.py', 62)],
			code: `class AttentionPairBias(nn.Module):
    def __init__(self, c_s=384, c_z=128, num_heads=16, initial_norm=True):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, c_s // num_heads  # 16 heads of width 24
        self.norm_s = nn.LayerNorm(c_s) if initial_norm else nn.Identity()
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)                # gate
        self.proj_z = nn.Sequential(nn.LayerNorm(c_z), nn.Linear(c_z, num_heads, bias=False))
        self.proj_o = nn.Linear(c_s, c_s, bias=False)

    def split_heads(self, t):                                        # (B, L, 384) -> (B, 16, L, 24)
        return t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def forward(self, s, z):                                         # s: (B, L, 384), z: (B, L, L, 128)
        s = self.norm_s(s)
        q = self.split_heads(self.proj_q(s))
        k = self.split_heads(self.proj_k(s))
        v = self.split_heads(self.proj_v(s))

        bias = self.proj_z(z).permute(0, 3, 1, 2)                    # (B, 16, L, L)
        logits = q @ k.transpose(-1, -2) / self.head_dim**0.5 + bias
        o = logits.softmax(dim=-1) @ v                               # (B, 16, L, 24)
        o = o.transpose(1, 2).flatten(-2)                            # (B, L, 384)
        return self.proj_o(self.proj_g(s).sigmoid() * o)


# In the block:
s = s + attention(s, z)`
		},
		{
			id: 'transition_s',
			title: 'Single transition',
			subtitle: '384 -> 1536 -> 384, SwiGLU',
			kind: 'seq',
			in: 's (L, 384)',
			out: 's update (L, 384)',
			summary: 'A gated feed-forward layer applied to every token independently. Its output, added to s, is the single state that leaves the block.',
			parent: 'block',
			position: { x: IN_BLOCK.s, y: 720 },
			visual: {
				lanes: [S],
				steps: [
					{ lane: 's', label: 'LayerNorm' },
					{ lane: 's', label: 'Two projections: Linear 384 -> 1536, no bias', note: 'SiLU on the first, multiply by the second' },
					{ lane: 's', label: 'Linear 1536 -> 384, no bias', residual: true }
				]
			},
			math: [{ tex: r`s_i \leftarrow s_i + W_3\Big(\mathrm{SiLU}\big(W_1 \hat s_i\big) \odot W_2 \hat s_i\Big), \qquad \hat s = \mathrm{LN}(s)` }],
			sources: [src('Transition', 'layers/transition.py', 8)],
			code: `${TRANSITION}

transition_s = Transition(dim=384, hidden=1536)
s = s + transition_s(s)                                              # (B, L, 384)`
		},

		// ---- After the trunk ----
		{
			id: 'distogram',
			title: 'Distogram head',
			subtitle: 'Linear 128 -> 64, symmetrised',
			kind: 'head',
			in: 'z (L, L, 128)',
			out: 'logits (L, L, 64)',
			summary: 'Predicts a distribution over distance bins for every token pair from the final pairwise state. Here z is made symmetric before the linear layer, not after.',
			position: { x: COL.far, y: AFTER },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Add transpose', note: 'z[i,j] + z[j,i]' },
					{ lane: 'z', label: 'Linear 128 -> 64', shape: '(L, L, 64)' }
				]
			},
			math: [{ tex: r`\ell_{ij} = W\big(z_{ij} + z_{ji}\big) + b` }],
			sources: [src('DistogramModule', 'modules/trunk.py', 656)],
			code: `distogram = nn.Linear(128, 64)                                       # DistogramModule.distogram

logits = distogram(z + z.transpose(1, 2))                            # (B, L, L, 64)`
		},
		{
			id: 'diffusion',
			title: 'Diffusion module: one denoising step',
			subtitle: 'x 200 sampling steps, the same weights every step',
			kind: 'struct',
			in: 's (L, 384)   z (L, L, 128)   s_inputs (L, 455)   noisy coordinates (A, 3)',
			out: 'atom coordinates (A, 3)',
			summary:
				'Replaces the AlphaFold2 structure module. Coordinates start as pure noise and are cleaned up step by step. At each step a network looks at the current noisy atoms and predicts where they should be, conditioned on the trunk outputs: an atom encoder pools atoms into tokens, a transformer over tokens uses the pair state as an attention bias, and an atom decoder turns the result back into a per-atom update. The trunk runs once; this network runs once per step. There are no frames or torsion angles: it works directly on atom positions.',
			position: { x: DIFF.x, y: DIFF.y },
			size: { width: DIFF.width, height: DIFF.height },
			visual: {
				lanes: [ATOMS, COND],
				repeat: 'x 200 sampling steps',
				steps: [
					{ lane: 'c', label: 'Single conditioning', note: 's, s_inputs and the noise level', shape: '(L, 768)' },
					{ lane: 'c', label: 'Pairwise conditioning', note: 'z and the relative position encoding', shape: '(L, L, 128)' },
					{ lane: 'x', label: 'Atom attention encoder', note: '3 blocks over atoms, pooled into tokens', shape: '(L, 768)' },
					{ lane: 'all', label: 'Token transformer', note: '24 layers, pair-biased attention' },
					{ lane: 'x', label: 'Atom attention decoder', note: '3 blocks over atoms', shape: 'update (A, 3)' },
					{ lane: 'x', label: 'Denoise and step', note: 'combine with the noisy input, move to a lower noise level' }
				]
			},
			math: [
				{ label: 'Network', tex: String.raw`F_\theta(x, \sigma) = \mathrm{Decoder}\Big(\mathrm{Transformer}\big(\mathrm{Encoder}(x),\ \tilde s(\sigma),\ \tilde z\big)\Big)` },
				{ label: 'Denoiser', tex: String.raw`D_\theta(x; \sigma) = c_{\mathrm{skip}}(\sigma)\, x + c_{\mathrm{out}}(\sigma)\, F_\theta\big(c_{\mathrm{in}}(\sigma)\, x,\ \sigma\big)` }
			],
			sources: [src('DiffusionModule', 'modules/diffusion.py', 41), src('DiffusionModule.forward', 'modules/diffusion.py', 168), src('AtomDiffusion', 'modules/diffusion.py', 284)],
			code: `class DiffusionModule(nn.Module):
    """The score model: called once per sampling step."""

    def __init__(self, token_s=384, token_z=128, atom_s=128, atom_z=16):
        super().__init__()
        self.single_conditioner = SingleConditioning(token_s)
        self.pairwise_conditioner = PairwiseConditioning(token_z, dim_token_rel_pos_feats=token_z)
        self.atom_attention_encoder = AtomAttentionEncoder(atom_s, atom_z, token_s, token_z, structure_prediction=True)
        self.s_to_a_linear = nn.Sequential(nn.LayerNorm(2 * token_s), nn.Linear(2 * token_s, 2 * token_s, bias=False))
        self.token_transformer = DiffusionTransformer(depth=24, dim=2 * token_s, dim_single_cond=2 * token_s, dim_pairwise=token_z)
        self.a_norm = nn.LayerNorm(2 * token_s)
        self.atom_attention_decoder = AtomAttentionDecoder(atom_s, atom_z, token_s)

    def forward(self, r_noisy, times, s_trunk, z_trunk, s_inputs, relative_position_encoding, feats):
        s, _ = self.single_conditioner(times, s_trunk, s_inputs)     # (B, L, 768)
        z = self.pairwise_conditioner(z_trunk, relative_position_encoding)   # (B, L, L, 128)

        a, q_skip, c_skip, p_skip = self.atom_attention_encoder(feats, s_trunk, z, r_noisy)   # a: (B, L, 768)
        a = a + self.s_to_a_linear(s)
        a = self.a_norm(self.token_transformer(a, s, z))
        return self.atom_attention_decoder(a, q_skip, c_skip, p_skip, feats)   # (B, A, 3)`
		},
		{
			id: 'single_cond',
			title: 'Single conditioning',
			subtitle: 'Trunk s + inputs + noise level',
			kind: 'seq',
			in: 's (L, 384)   s_inputs (L, 455)   noise level',
			out: 'conditioning s (L, 768)',
			summary:
				'Builds the per-token conditioning signal. The trunk single state and the token input features are concatenated and projected, and an embedding of the current noise level is added so the network knows how noisy its input is. Two transition layers follow. This is how the trunk\'s single state reaches the diffusion module.',
			parent: 'diffusion',
			position: { x: IN_DIFF.s, y: 50 },
			visual: {
				lanes: [{ id: 's', label: 'per token', kind: 'seq' }],
				steps: [
					{ lane: 's', label: 'Concatenate trunk s and s_inputs', shape: '(L, 839)' },
					{ lane: 's', label: 'LayerNorm, Linear 839 -> 768', shape: '(L, 768)' },
					{ lane: 's', label: 'Fourier embedding of the noise level', note: 'cos of a frozen random projection, LayerNorm, Linear 256 -> 768; the same for every token', residual: true },
					{ lane: 's', label: '2 transitions', note: '768 -> 1536 -> 768, SwiGLU', residual: true }
				]
			},
			math: [
				{ label: 'Noise embedding', tex: String.raw`n = \mathrm{LN}\Big(\cos\big(2\pi\,(w\, c_{\mathrm{noise}}(\sigma) + b)\big)\Big) \in \mathbb{R}^{256}, \qquad c_{\mathrm{noise}}(\sigma) = \tfrac14 \ln\frac{\sigma}{\sigma_{\mathrm{data}}}` },
				{ label: 'Conditioning', tex: String.raw`\tilde s_i = W\,\mathrm{LN}\big([s_i \,\Vert\, s^{\mathrm{in}}_i]\big) + b + W_n\, n, \qquad \tilde s \leftarrow \tilde s + \mathrm{Transition}(\tilde s) \ \ (\times 2)` }
			],
			sources: [src('SingleConditioning', 'modules/encoders.py', 136), src('FourierEmbedding', 'modules/encoders.py', 18)],
			code: `${TRANSITION}

class FourierEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)                              # random features, never trained

    def forward(self, times):                                        # (B,) one noise level per sample
        return torch.cos(2 * torch.pi * self.proj(times[:, None]))   # (B, 256)


class SingleConditioning(nn.Module):
    def __init__(self, token_s=384, s_input_dim=455, dim_fourier=256, num_transitions=2):
        super().__init__()
        input_dim = token_s + s_input_dim                            # 839
        self.norm_single = nn.LayerNorm(input_dim)
        self.single_embed = nn.Linear(input_dim, 2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = nn.Linear(dim_fourier, 2 * token_s, bias=False)
        self.transitions = nn.ModuleList(Transition(2 * token_s, 4 * token_s) for _ in range(num_transitions))

    def forward(self, times, s_trunk, s_inputs):                     # (B,), (B, L, 384), (B, L, 455)
        s = torch.cat([s_trunk, s_inputs], dim=-1)
        s = self.single_embed(self.norm_single(s))                   # (B, L, 768)

        fourier = self.norm_fourier(self.fourier_embed(times))       # (B, 256)
        s = s + self.fourier_to_single(fourier)[:, None]             # the same vector added to every token

        for transition in self.transitions:
            s = s + transition(s)
        return s, fourier`
		},
		{
			id: 'pair_cond',
			title: 'Pairwise conditioning',
			subtitle: 'Trunk z + relative position',
			kind: 'pair',
			in: 'z (L, L, 128)   relative position encoding (L, L, 128)',
			out: 'conditioning z (L, L, 128)',
			summary:
				'Builds the per-pair conditioning signal from the trunk pairwise state and the relative position encoding. It does not depend on the noise level, so it is computed once and reused for all sampling steps. This is how the trunk\'s pairwise state reaches the diffusion module: as the attention bias in the token transformer and as pair features for the atom transformers.',
			parent: 'diffusion',
			position: { x: IN_DIFF.z, y: 50 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Concatenate trunk z and the relative position encoding', shape: '(L, L, 256)' },
					{ lane: 'z', label: 'LayerNorm, Linear 256 -> 128, no bias', shape: '(L, L, 128)' },
					{ lane: 'z', label: '2 transitions', note: '128 -> 256 -> 128, SwiGLU', residual: true }
				]
			},
			math: [{ tex: String.raw`\tilde z_{ij} = W\,\mathrm{LN}\big([z_{ij} \,\Vert\, \mathrm{relpos}_{ij}]\big), \qquad \tilde z \leftarrow \tilde z + \mathrm{Transition}(\tilde z) \ \ (\times 2)` }],
			sources: [src('PairwiseConditioning', 'modules/encoders.py', 209)],
			code: `${TRANSITION}

class PairwiseConditioning(nn.Module):
    def __init__(self, token_z=128, dim_token_rel_pos_feats=128, num_transitions=2):
        super().__init__()
        dim = token_z + dim_token_rel_pos_feats
        self.dim_pairwise_init_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, token_z, bias=False))
        self.transitions = nn.ModuleList(Transition(token_z, 2 * token_z) for _ in range(num_transitions))

    def forward(self, z_trunk, token_rel_pos_feats):                 # (B, L, L, 128) each
        z = self.dim_pairwise_init_proj(torch.cat([z_trunk, token_rel_pos_feats], dim=-1))
        for transition in self.transitions:
            z = z + transition(z)
        return z                                                     # (B, L, L, 128)`
		},
		{
			id: 'atom_encoder',
			title: 'Atom attention encoder',
			subtitle: '3 blocks over atoms, pooled to tokens',
			kind: 'struct',
			in: 'noisy coordinates (A, 3)   atom features   conditioning',
			out: 'token activations a (L, 768)',
			summary:
				'Reads the noisy structure at atom resolution. Each atom starts from an embedding of its chemistry, to which the trunk single state of its token and a projection of its current noisy position are added. Atoms then attend to nearby atoms in the sequence (windows of 32 queries seeing 128 keys), with atom-pair features built from the reference conformer and the pair conditioning. Finally the atoms of each token are averaged into one vector, so the expensive token transformer runs on L tokens rather than A atoms.',
			parent: 'diffusion',
			position: { x: IN_DIFF.mid, y: 220 },
			visual: {
				lanes: [ATOMS, COND],
				steps: [
					{ lane: 'x', label: 'Embed atom features', note: 'reference position, charge, element, atom name: Linear -> 128' },
					{ lane: 'all', label: 'Add the trunk single state of each atom\'s token', note: 'LayerNorm, Linear 384 -> 128, broadcast token -> atoms' },
					{ lane: 'all', label: 'Atom-pair features', note: 'reference-conformer offsets and distances, plus the pair conditioning broadcast to atom pairs', shape: '(windows, 32, 128, 16)' },
					{ lane: 'x', label: 'Add the noisy position: Linear 10 -> 128', note: 'x, y, z padded with zeros' },
					{ lane: 'x', label: 'Atom transformer', note: '3 layers, 4 heads, attention within local windows' },
					{ lane: 'x', label: 'Linear 128 -> 768, ReLU, mean over each token\'s atoms', shape: '(L, 768)' }
				]
			},
			math: [
				{ label: 'Atom inputs', tex: String.raw`c_l = W_f f_l + W_s\,\mathrm{LN}\big(s_{\mathrm{tok}(l)}\big), \qquad q_l = c_l + W_r\, [\,\vec x_l \,\Vert\, \vec 0\,]` },
				{ label: 'Atom-pair features', tex: String.raw`p_{lm} = v_{lm}\Big(W_d\, \vec d_{lm} + W_n \tfrac{1}{1 + \lVert \vec d_{lm} \rVert^2} + w_v\Big) + W_z\,\mathrm{LN}\big(\tilde z_{\mathrm{tok}(l)\,\mathrm{tok}(m)}\big) + \dots`, note: 'd is the offset between two atoms in the reference conformer, and v is 1 only for atoms of the same residue.' },
				{ label: 'Pool to tokens', tex: String.raw`a_i = \operatorname{mean}_{l:\ \mathrm{tok}(l) = i}\ \mathrm{ReLU}\big(W q_l\big)` }
			],
			sources: [src('AtomAttentionEncoder', 'modules/encoders.py', 288), src('AtomAttentionEncoder.forward', 'modules/encoders.py', 395), src('AtomTransformer', 'modules/transformers.py', 252)],
			code: `# Outline of AtomAttentionEncoder.forward. The windowing that restricts each atom to nearby
# keys is left out; see get_indexing_matrix and single_to_keys in the source.
def atom_attention_encoder(feats, s_trunk, z, r_noisy):
    atom_feats = torch.cat(
        [feats["ref_pos"], feats["ref_charge"][..., None], feats["atom_pad_mask"][..., None],
         feats["ref_element"], feats["ref_atom_name_chars"].flatten(-2)],
        dim=-1,
    )
    c = embed_atom_features(atom_feats)                              # nn.Linear(389, 128, bias=False) -> (B, A, 128)

    # Atom-pair features from the reference conformer, only between atoms of the same residue.
    d = ref_pos_keys - ref_pos_queries                               # (B, windows, 32, 128, 3)
    p = (embed_atompair_ref_pos(d) + embed_atompair_ref_dist(1 / (1 + d.pow(2).sum(-1, keepdim=True)))
         + embed_atompair_mask(v)) * v                               # (B, windows, 32, 128, 16)

    # Condition on the trunk: single state per atom, pair conditioning per atom pair.
    atom_to_token = feats["atom_to_token"].float()                   # (B, A, L) one-hot
    c = c + atom_to_token @ s_to_c_trans(s_trunk)                    # LayerNorm, nn.Linear(384, 128, bias=False)
    p = p + broadcast_to_atom_pairs(z_to_p_trans(z))                 # LayerNorm, nn.Linear(128, 16, bias=False)
    p = p + c_to_p_trans_q(c_queries) + c_to_p_trans_k(c_keys)
    p = p + p_mlp(p)

    q = c + r_to_q_trans(F.pad(r_noisy, (0, 7)))                     # nn.Linear(10, 128, bias=False)
    q = atom_encoder(q, c, p)                                        # 3 layers, 4 heads, windowed

    # Mean over the atoms of each token.
    a = atom_to_token_trans(q)                                       # nn.Linear(128, 768, bias=False), ReLU
    weights = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
    a = weights.transpose(1, 2) @ a                                  # (B, L, 768)
    return a, q, c, p                                                # q, c, p are skip connections for the decoder`
		},
		{
			id: 'token_transformer',
			title: 'Token transformer',
			subtitle: 'x 24 layers, pair-biased, noise-conditioned',
			kind: 'cross',
			in: 'a (L, 768)   conditioning s (L, 768)   conditioning z (L, L, 128)',
			out: 'a (L, 768)',
			summary:
				'The main network of the diffusion module: a transformer over tokens. Its attention is the same pair-biased attention as in the Pairformer, with the pair conditioning as the bias. The single conditioning enters through adaptive LayerNorm, which sets a per-token scale and shift, and through sigmoid gates on each sub-layer\'s output. Unlike the Pairformer there are no triangular updates: the pair information is fixed and only read.',
			parent: 'diffusion',
			position: { x: IN_DIFF.mid, y: 390 },
			visual: {
				lanes: [TOKENS, COND],
				repeat: 'x 24 layers',
				steps: [
					{ lane: 'all', label: 'Adaptive LayerNorm', note: 'normalise a, then scale and shift by projections of the single conditioning' },
					{ lane: 'all', label: 'Attention with pair bias', note: 'bias from the pair conditioning; no LayerNorm of its own' },
					{ lane: 'all', label: 'Gate by sigmoid(Linear(conditioning s))', residual: true },
					{ lane: 'all', label: 'Conditioned transition', note: 'adaptive LayerNorm, SwiGLU 768 -> 1536 -> 768, gated the same way', residual: true }
				]
			},
			math: [
				{ label: 'Adaptive LayerNorm', tex: String.raw`\mathrm{AdaLN}(a, \tilde s) = \sigma\big(W_\gamma\,\mathrm{LN}(\tilde s) + b_\gamma\big) \odot \mathrm{LN}(a) + W_\beta\,\mathrm{LN}(\tilde s)` },
				{ label: 'Attention sub-layer', tex: String.raw`a \leftarrow a + \sigma\big(W_g \tilde s + b_g\big) \odot \mathrm{AttnPairBias}\big(\mathrm{AdaLN}(a, \tilde s),\ \mathrm{bias} = \tilde z\big)` },
				{ label: 'Transition sub-layer', tex: String.raw`a \leftarrow a + \sigma\big(W_{g'} \tilde s + b_{g'}\big) \odot W_3\Big(\mathrm{SiLU}(W_1 \hat a) \odot W_2 \hat a \odot W_4 \hat a\Big), \qquad \hat a = \mathrm{AdaLN}(a, \tilde s)` }
			],
			sources: [
				src('DiffusionTransformerLayer', 'modules/transformers.py', 180),
				src('AdaLN', 'modules/transformers.py', 17),
				src('ConditionedTransitionBlock', 'modules/transformers.py', 44),
				src('AttentionPairBias', 'layers/attention.py', 8)
			],
			code: `class AdaLN(nn.Module):
    def __init__(self, dim, dim_single_cond):
        super().__init__()
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False)    # no learned scale or shift of its own
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = nn.Linear(dim_single_cond, dim, bias=False)

    def forward(self, a, s):
        s = self.s_norm(s)
        return torch.sigmoid(self.s_scale(s)) * self.a_norm(a) + self.s_bias(s)


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, dim, dim_single_cond, expansion_factor=2):
        super().__init__()
        inner = dim * expansion_factor
        self.adaln = AdaLN(dim, dim_single_cond)
        self.swish_gate = nn.Linear(dim, 2 * inner, bias=False)      # followed by SwiGLU in the source
        self.a_to_b = nn.Linear(dim, inner, bias=False)
        self.b_to_a = nn.Linear(inner, dim, bias=False)
        self.output_projection = nn.Sequential(nn.Linear(dim_single_cond, dim), nn.Sigmoid())

    def forward(self, a, s):
        a = self.adaln(a, s)
        x, gate = self.swish_gate(a).chunk(2, dim=-1)
        b = F.silu(gate) * x * self.a_to_b(a)
        return self.output_projection(s) * self.b_to_a(b)


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, heads, dim=768, dim_single_cond=768, dim_pairwise=128):
        super().__init__()
        self.adaln = AdaLN(dim, dim_single_cond)
        # The Pairformer's attention class, without its own LayerNorm: AdaLN has already normalised.
        self.pair_bias_attn = AttentionPairBias(dim, dim_pairwise, heads, initial_norm=False)
        self.output_projection = nn.Sequential(nn.Linear(dim_single_cond, dim), nn.Sigmoid())
        self.transition = ConditionedTransitionBlock(dim, dim_single_cond)

    def forward(self, a, s, z):                                      # (B, L, 768), (B, L, 768), (B, L, L, 128)
        b = self.pair_bias_attn(self.adaln(a, s), z)
        a = a + self.output_projection(s) * b
        return a + self.transition(a, s)


# DiffusionTransformer is 24 of these in sequence (depth and heads are set by the checkpoint).
for layer in token_transformer.layers:
    a = layer(a, s, z)`
		},
		{
			id: 'atom_decoder',
			title: 'Atom attention decoder',
			subtitle: '3 blocks over atoms, Linear to xyz',
			kind: 'struct',
			in: 'a (L, 768)   atom skip connections',
			out: 'coordinate update (A, 3)',
			summary:
				'Goes back from tokens to atoms. Each token\'s activation is projected and added to the encoder\'s per-atom features for all atoms of that token. Three more layers of windowed atom attention let neighbouring atoms coordinate, and a final linear layer turns each atom\'s features into a 3D vector. That vector is the network output: a raw update, not yet a position.',
			parent: 'diffusion',
			position: { x: IN_DIFF.mid, y: 560 },
			visual: {
				lanes: [ATOMS, TOKENS],
				steps: [
					{ lane: 'a', label: 'Linear 768 -> 128, no bias' },
					{ lane: 'all', label: 'Broadcast each token to its atoms, add to the encoder\'s atom features', residual: true },
					{ lane: 'x', label: 'Atom transformer', note: '3 layers, 4 heads, the same windows and pair features as the encoder' },
					{ lane: 'x', label: 'LayerNorm, Linear 128 -> 3, no bias', shape: '(A, 3)' }
				]
			},
			math: [
				{ label: 'Tokens to atoms', tex: String.raw`q_l \leftarrow q^{\mathrm{skip}}_l + W_a\, a_{\mathrm{tok}(l)}` },
				{ label: 'Update', tex: String.raw`\vec r_l = W_r\,\mathrm{LN}\big(\mathrm{AtomTransformer}(q, c^{\mathrm{skip}}, p^{\mathrm{skip}})_l\big) \in \mathbb{R}^3` }
			],
			sources: [src('AtomAttentionDecoder', 'modules/encoders.py', 543), src('AtomAttentionDecoder.forward', 'modules/encoders.py', 600), src('AtomTransformer', 'modules/transformers.py', 252)],
			code: `# Outline of AtomAttentionDecoder.forward, without the windowing.
a_to_q_trans = nn.Linear(768, 128, bias=False)
atom_feat_to_atom_pos_update = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 3, bias=False))


def atom_attention_decoder(a, q_skip, c_skip, p_skip, feats):
    atom_to_token = feats["atom_to_token"].float()                   # (B, A, L) one-hot
    q = q_skip + atom_to_token @ a_to_q_trans(a)                     # every atom gets its token's vector
    q = atom_decoder(q, c_skip, p_skip)                              # 3 layers, 4 heads, windowed
    return atom_feat_to_atom_pos_update(q)                           # (B, A, 3)`
		},
		{
			id: 'sampler',
			title: 'Denoise and step',
			subtitle: 'EDM preconditioning, 200 steps',
			kind: 'struct',
			in: 'noisy coordinates (A, 3)   update (A, 3)   noise level',
			out: 'coordinates at the next, lower noise level (A, 3)',
			summary:
				'The sampling loop around the network. The network\'s raw output is blended with its noisy input, with weights that depend on the noise level, to give an estimate of the clean structure. The coordinates then move part of the way from the noisy input towards that estimate, to the next noise level on a fixed schedule from very high noise down to zero. Before each step the structure is randomly rotated and shifted, since the network is not built to be rotation-invariant, and a little fresh noise is added.',
			parent: 'diffusion',
			position: { x: IN_DIFF.mid, y: 730 },
			visual: {
				lanes: [ATOMS],
				steps: [
					{ lane: 'x', label: 'Start: Gaussian noise at the highest noise level', note: 'standard deviation 160 x 16 angstroms', shape: '(A, 3)' },
					{ lane: 'x', label: 'Centre, apply a random rotation and translation' },
					{ lane: 'x', label: 'Add a little noise', note: 'raise the noise level by a factor 1.8 while it is above 1' },
					{ lane: 'x', label: 'Denoise', note: 'c_skip * noisy + c_out * network(c_in * noisy)' },
					{ lane: 'x', label: 'Step towards the denoised estimate', note: 'step scale 1.5, down to the next noise level' }
				],
				repeat: 'the last four steps, x 200'
			},
			math: [
				{ label: 'Noise schedule', tex: String.raw`\sigma_t = \sigma_{\mathrm{data}} \Big(\sigma_{\max}^{1/\rho} + \tfrac{t}{N - 1}\big(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\big)\Big)^{\rho}, \qquad t = 0, \dots, N - 1, \qquad \sigma_N = 0`, note: 'sigma_data = 16, sigma_max = 160, sigma_min = 0.0004, rho = 7, N = 200.' },
				{
					label: 'Preconditioning',
					tex: String.raw`c_{\mathrm{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_{\mathrm{data}}^2}}, \qquad c_{\mathrm{skip}} = \frac{\sigma_{\mathrm{data}}^2}{\sigma^2 + \sigma_{\mathrm{data}}^2}, \qquad c_{\mathrm{out}} = \frac{\sigma\, \sigma_{\mathrm{data}}}{\sqrt{\sigma^2 + \sigma_{\mathrm{data}}^2}}`
				},
				{ label: 'Add noise', tex: String.raw`\hat\sigma = \sigma_t (1 + \gamma), \qquad \hat x = x_t + \lambda \sqrt{\hat\sigma^2 - \sigma_t^2}\ \epsilon, \qquad \epsilon \sim \mathcal N(0, I)`, note: 'gamma = 0.8 while sigma is above 1, otherwise 0. lambda = 1.003.' },
				{ label: 'Step', tex: String.raw`x_{t+1} = \hat x + \eta\, (\sigma_{t+1} - \hat\sigma)\, \frac{\hat x - D_\theta(\hat x; \hat\sigma)}{\hat\sigma}, \qquad \eta = 1.5` }
			],
			sources: [
				src('AtomDiffusion.preconditioned_network_forward', 'modules/diffusion.py', 404),
				src('AtomDiffusion.sample_schedule', 'modules/diffusion.py', 430),
				src('AtomDiffusion.sample', 'modules/diffusion.py', 449)
			],
			code: `sigma_data, sigma_min, sigma_max, rho = 16.0, 0.0004, 160.0, 7
gamma_0, gamma_min, noise_scale, step_scale = 0.8, 1.0, 1.003, 1.5


def denoise(score_model, x_noisy, sigma, **conditioning):            # x_noisy: (B, A, 3), sigma: float
    c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    times = torch.full((x_noisy.shape[0],), 0.25 * math.log(sigma / sigma_data))
    r_update = score_model(r_noisy=c_in * x_noisy, times=times, **conditioning)
    return c_skip * x_noisy + c_out * r_update


def sample_schedule(num_steps=200):
    steps = torch.arange(num_steps) / (num_steps - 1)
    sigmas = (sigma_max ** (1 / rho) + steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return F.pad(sigmas * sigma_data, (0, 1))                        # ends at exactly 0


@torch.no_grad()
def sample(score_model, num_atoms, **conditioning):
    sigmas = sample_schedule()
    gammas = torch.where(sigmas > gamma_min, gamma_0, 0.0)

    x = sigmas[0] * torch.randn(1, num_atoms, 3)                     # pure noise
    for sigma_prev, sigma_next, gamma in zip(sigmas[:-1].tolist(), sigmas[1:].tolist(), gammas[1:].tolist()):
        rotation, shift = random_rotation_and_translation()
        x = (x - x.mean(dim=-2, keepdim=True)) @ rotation + shift

        t_hat = sigma_prev * (1 + gamma)                             # briefly raise the noise level
        x_noisy = x + noise_scale * (t_hat**2 - sigma_prev**2) ** 0.5 * torch.randn_like(x)

        x_denoised = denoise(score_model, x_noisy, t_hat, **conditioning)
        x = x_noisy + step_scale * (sigma_next - t_hat) * (x_noisy - x_denoised) / t_hat
    return x                                                         # (1, A, 3)`
		},
		{
			id: 'confidence',
			title: 'Confidence module',
			subtitle: 'Its own Pairformer + heads',
			kind: 'head',
			in: 's, z, s_inputs, predicted coordinates',
			out: 'pLDDT (L,)   PDE, PAE (L, L)',
			summary:
				'Scores the predicted structure. The trunk outputs are combined with an embedding of the distances in the predicted coordinates, refined by a separate Pairformer stack, and read out by linear heads for per-token lDDT, pairwise distance error, pairwise aligned error and whether each token is resolved. All inputs are detached, so it does not train the trunk.',
			position: { x: COL.mid, y: BELOW + 70 },
			visual: {
				lanes: [S, Z],
				steps: [
					{ lane: 's', label: 'LayerNorm of s, plus a projection of s_inputs' },
					{ lane: 'z', label: 'LayerNorm of z, plus pair sums of s_inputs' },
					{ lane: 'z', label: 'Embedding of binned predicted distances', residual: true },
					{ lane: 'all', label: 'Pairformer stack', note: 'separate weights from the trunk' },
					{ lane: 's', label: 'Linear -> pLDDT bins, Linear -> resolved', note: 'no bias' },
					{ lane: 'z', label: 'Linear -> PDE bins, Linear -> PAE bins', note: 'no bias; PDE uses z + z transposed' }
				]
			},
			math: [
				{ label: 'Distance conditioning', tex: r`z_{ij} \leftarrow z_{ij} + E_{\mathrm{dist}}\big[\operatorname{bin}(\lVert \hat x_i - \hat x_j \rVert)\big]` },
				{ label: 'Heads', tex: r`\mathrm{pLDDT}_i = \mathbb{E}\big[\operatorname{softmax}(W_{\mathrm{lddt}}\, s_i)\big], \qquad \mathrm{PAE}_{ij} = \mathbb{E}\big[\operatorname{softmax}(W_{\mathrm{pae}}\, z_{ij})\big]` }
			],
			sources: [src('ConfidenceModule', 'modules/confidence.py', 20), src('ConfidenceModule.forward', 'modules/confidence.py', 183), src('ConfidenceHeads', 'modules/confidence.py', 337)],
			code: `# Outline. The exact conditioning depends on checkpoint flags; see ConfidenceModule.__init__.
def confidence_module(s_inputs, s, z, x_pred):                       # all inputs detached
    s = s_norm(s) + s_input_to_s(s_inputs_norm(s_inputs))
    z = z_norm(z)
    z = z + s_to_z(s_inputs)[:, :, None] + s_to_z_transpose(s_inputs)[:, None, :]
    z = z + dist_bin_pairwise_embed(distance_bins(x_pred))           # nn.Embedding(num_dist_bins, 128)

    s, z = pairformer_stack(s, z)                                    # separate weights from the trunk

    return {
        "plddt_logits": to_plddt_logits(s),                          # nn.Linear(384, bins, bias=False)
        "resolved_logits": to_resolved_logits(s),                    # nn.Linear(384, 2, bias=False)
        "pde_logits": to_pde_logits(z + z.transpose(1, 2)),          # nn.Linear(128, bins, bias=False)
        "pae_logits": to_pae_logits(z),                              # nn.Linear(128, bins, bias=False)
    }`
		}
	],

	tensors: [
		{
			id: 't_atoms', label: 'Atom features', shape: '(A, ...)', kind: 'input', position: { x: COL.s + T, y: 0 },
			description: 'Every heavy atom in the input: its element, charge, name and position in an idealised reference conformer of its residue. This is chemistry, not the answer: the reference conformer says nothing about how the chain folds.',
			axes: [{ dim: 'A', meaning: 'atoms in the complex' }, { dim: '...', meaning: 'several feature tensors per atom' }]
		},
		{
			id: 't_tok', label: 'Token features', shape: '(L, 71)', kind: 'input', position: { x: COL.mid + T, y: 0 },
			description: 'Per-token features. For a protein, one token is one residue. Includes the residue type and a summary of the MSA at that position.',
			axes: [{ dim: 'L', meaning: 'tokens (residues, for proteins)' }, { dim: '71', meaning: 'residue type 33 + MSA profile 33 + deletion mean 1 + pocket flag 4' }]
		},
		{
			id: 't_rel', label: 'Positions and bonds', shape: '(L, L, ...)', kind: 'input', position: { x: COL.z + T, y: 0 },
			description: 'Pairwise input features: how far apart two tokens are along the chain, whether they are in the same chain, and whether they are covalently bonded.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of tokens (i, j)' }, { dim: '...', meaning: 'relative position one-hots (139) and a bond flag (1)' }]
		},
		{
			id: 't_sin', label: 's_inputs', shape: '(L, 455)', kind: 'input', position: { x: COL.mid + T, y: 180 },
			description: 'The fixed per-token input vector. It is never updated, and is used again by the MSA module, the diffusion module and the confidence module.',
			axes: [{ dim: 'L', meaning: 'tokens' }, { dim: '455', meaning: 'atom summary 384 + token features 71' }]
		},
		{
			id: 't_sinit', label: 's_init', shape: '(L, 384)', kind: 'seq', position: { x: COL.s + T, y: 350 },
			description: 'The initial single state: a linear projection of s_inputs.',
			axes: [{ dim: 'L', meaning: 'tokens' }, { dim: '384', meaning: 'single width' }]
		},
		{
			id: 't_zinit', label: 'z_init', shape: '(L, L, 128)', kind: 'pair', position: { x: COL.z + T, y: 350 },
			description: 'The initial pairwise state. It already encodes which residue types tokens i and j are, their relative position and any bond between them.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of tokens (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_s_in', label: 'Single state s', shape: '(L, 384)', kind: 'seq', position: { x: COL.s + T, y: 520 },
			description: 'The single state entering the Pairformer: one vector per token. For the first block it is s_init plus the recycling term. For later blocks it is the previous block\'s output.',
			axes: [{ dim: 'L', meaning: 'tokens' }, { dim: '384', meaning: 'single width' }]
		},
		{
			id: 't_z_rec', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', position: { x: COL.z + T, y: 520 },
			description: 'The pairwise state after recycling and before the MSA module has added alignment information.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of tokens (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_msa', label: 'MSA features', shape: '(S, L, 35)', kind: 'input', position: { x: COL.far + T, y: 520 },
			description: 'The multiple sequence alignment: related sequences aligned to the target, as a one-hot residue type plus deletion features. With the paired-chain flag used for complexes the width is 36.',
			axes: [{ dim: 'S', meaning: 'aligned sequences' }, { dim: 'L', meaning: 'tokens' }, { dim: '35', meaning: 'residue one-hot 33 + has-deletion 1 + deletion value 1' }]
		},
		{
			id: 't_z_in', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', position: { x: COL.z + T, y: 690 },
			description: 'The pairwise state entering the Pairformer, with the MSA module\'s update added.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of tokens (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_z_out', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', parent: 'block', position: { x: IN_BLOCK.z + T, y: 550 },
			description: 'The pairwise state leaving the block. The single-state attention in the same block reads it as a bias. After the last block it goes to the distogram head, the diffusion module and the confidence module.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of tokens (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_s_out', label: 'Single state s', shape: '(L, 384)', kind: 'seq', parent: 'block', position: { x: IN_BLOCK.s + T, y: 820 },
			description: 'The single state leaving the block, after pair-biased attention and the transition.',
			axes: [{ dim: 'L', meaning: 'tokens' }, { dim: '384', meaning: 'single width' }]
		},
		{
			id: 't_disto', label: 'Distogram logits', shape: '(L, L, 64)', kind: 'head', position: { x: COL.far + T, y: AFTER + 100 },
			description: 'A predicted distribution over the distance between every pair of tokens, read directly from the final pairwise state.',
			axes: [{ dim: 'L, L', meaning: 'pair of tokens' }, { dim: '64', meaning: 'distance bins' }]
		},
		{
			id: 't_scond', label: 'Single conditioning', shape: '(L, 768)', kind: 'seq', parent: 'diffusion', position: { x: IN_DIFF.s + T, y: 150 },
			description: 'Per-token conditioning for the diffusion network: the trunk single state and input features, plus the current noise level. It changes every sampling step because the noise level does.',
			axes: [{ dim: 'L', meaning: 'tokens' }, { dim: '768', meaning: 'twice the trunk single width' }]
		},
		{
			id: 't_zcond', label: 'Pair conditioning', shape: '(L, L, 128)', kind: 'pair', parent: 'diffusion', position: { x: IN_DIFF.z + T, y: 150 },
			description: 'Per-pair conditioning: the trunk pairwise state combined with relative position. It is the same at every sampling step, and is the only way the diffusion network sees which tokens should be close together.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of tokens (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_xnoisy', label: 'Noisy coordinates', shape: '(A, 3)', kind: 'struct', parent: 'diffusion', position: { x: IN_DIFF.mid + T, y: 150 },
			description: 'The current guess for every atom position, with noise. At the first step this is pure Gaussian noise with no structure at all; each step\'s output becomes the next step\'s input at a lower noise level.',
			axes: [{ dim: 'A', meaning: 'atoms in the complex' }, { dim: '3', meaning: 'x, y, z in angstroms' }]
		},
		{
			id: 't_a', label: 'Token activations a', shape: '(L, 768)', kind: 'seq', parent: 'diffusion', position: { x: IN_DIFF.mid + T, y: 320 },
			description: 'One vector per token summarising where its atoms currently are in the noisy structure, pooled from the atom encoder.',
			axes: [{ dim: 'L', meaning: 'tokens' }, { dim: '768', meaning: 'diffusion token width' }]
		},
		{
			id: 't_a_out', label: 'Token activations a', shape: '(L, 768)', kind: 'seq', parent: 'diffusion', position: { x: IN_DIFF.mid + T, y: 490 },
			description: 'The token activations after the 24-layer transformer: each token now carries what it needs to tell its atoms where to move.',
			axes: [{ dim: 'L', meaning: 'tokens' }, { dim: '768', meaning: 'diffusion token width' }]
		},
		{
			id: 't_rupdate', label: 'Coordinate update', shape: '(A, 3)', kind: 'struct', parent: 'diffusion', position: { x: IN_DIFF.mid + T, y: 660 },
			description: 'The raw network output, one 3D vector per atom. It is not a position on its own: the sampler scales it and blends it with the noisy input to get the denoised estimate.',
			axes: [{ dim: 'A', meaning: 'atoms in the complex' }, { dim: '3', meaning: 'x, y, z' }]
		},
		{
			id: 't_coords', label: 'Atom coordinates', shape: '(A, 3)', kind: 'struct', position: { x: COL.mid + T, y: BELOW },
			description: 'The predicted structure: a 3D position for every heavy atom, sampled by the diffusion process. Different random seeds give different samples.',
			axes: [{ dim: 'A', meaning: 'atoms in the complex' }, { dim: '3', meaning: 'x, y, z in angstroms' }]
		},
		{
			id: 't_conf', label: 'pLDDT, PDE, PAE', shape: '(L,), (L, L), (L, L)', kind: 'head', position: { x: COL.mid + T, y: BELOW + 170 },
			description: 'Confidence estimates for the sampled structure: per-token local accuracy, and pairwise errors in distance and in aligned position. pTM and ipTM are computed from the PAE.',
			axes: [{ dim: '(L,)', meaning: 'pLDDT per token' }, { dim: '(L, L)', meaning: 'predicted distance error and predicted aligned error per token pair' }]
		}
	],

	edges: [
		{ from: 't_atoms', to: 'input_embedder', kind: 'input' },
		{ from: 't_tok', to: 'input_embedder', kind: 'input' },
		{ from: 'input_embedder', to: 't_sin', kind: 'input' },
		{ from: 't_sin', to: 's_init', kind: 'input' },
		{ from: 't_sin', to: 'z_init', kind: 'input' },
		{ from: 't_rel', to: 'z_init', kind: 'input' },
		{ from: 's_init', to: 't_sinit', kind: 'seq' },
		{ from: 'z_init', to: 't_zinit', kind: 'pair' },
		{ from: 't_sinit', to: 'recycling', kind: 'seq' },
		{ from: 't_zinit', to: 'recycling', kind: 'pair' },
		{ from: 'recycling', to: 't_s_in', kind: 'seq' },
		{ from: 'recycling', to: 't_z_rec', kind: 'pair' },
		{ from: 't_z_rec', to: 'msa_module', kind: 'pair' },
		{ from: 't_msa', to: 'msa_module', kind: 'input' },
		{ from: 'msa_module', to: 't_z_in', kind: 'pair' },

		{ from: 't_z_in', to: 'tri_mul_out', kind: 'pair' },
		{ from: 'tri_mul_out', to: 'tri_mul_in', kind: 'pair' },
		{ from: 'tri_mul_in', to: 'tri_att_start', kind: 'pair' },
		{ from: 'tri_att_start', to: 'tri_att_end', kind: 'pair' },
		{ from: 'tri_att_end', to: 'transition_z', kind: 'pair' },
		{ from: 'transition_z', to: 't_z_out', kind: 'pair' },
		{ from: 't_z_out', to: 'attention', label: 'pair bias', kind: 'cross' },
		{ from: 't_s_in', to: 'attention', kind: 'seq' },
		{ from: 'attention', to: 'transition_s', kind: 'seq' },
		{ from: 'transition_s', to: 't_s_out', kind: 'seq' },

		{ from: 't_z_out', to: 'distogram', kind: 'pair' },
		{ from: 'distogram', to: 't_disto', kind: 'head' },
		{ from: 't_s_out', to: 'single_cond', kind: 'seq' },
		{ from: 't_z_out', to: 'pair_cond', kind: 'pair' },
		{ from: 'single_cond', to: 't_scond', kind: 'seq' },
		{ from: 'pair_cond', to: 't_zcond', kind: 'pair' },
		{ from: 't_xnoisy', to: 'atom_encoder', kind: 'struct' },
		{ from: 't_zcond', to: 'atom_encoder', kind: 'pair' },
		{ from: 'atom_encoder', to: 't_a', kind: 'seq' },
		{ from: 't_a', to: 'token_transformer', kind: 'seq' },
		{ from: 't_scond', to: 'token_transformer', kind: 'seq' },
		{ from: 't_zcond', to: 'token_transformer', label: 'pair bias', kind: 'pair' },
		{ from: 'token_transformer', to: 't_a_out', kind: 'seq' },
		{ from: 't_a_out', to: 'atom_decoder', kind: 'seq' },
		{ from: 'atom_decoder', to: 't_rupdate', kind: 'struct' },
		{ from: 't_rupdate', to: 'sampler', kind: 'struct' },
		{ from: 'sampler', to: 't_coords', kind: 'struct' },
		{ from: 't_coords', to: 'confidence', kind: 'struct' },
		{ from: 'confidence', to: 't_conf', kind: 'head' }
	]
};
