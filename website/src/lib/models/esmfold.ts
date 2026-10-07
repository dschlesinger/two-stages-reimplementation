import { structureModule } from './structure.ts';
import type { Lane, ModelSpec } from './types';

// Dimensions and operation order follow transformers' modeling_esmfold.py
// (EsmForProteinFolding, facebook/esmfold_v1 config).

const r = String.raw;

// Reference implementation, pinned to one commit so the line anchors stay valid.
const REPO = 'https://github.com/huggingface/transformers/blob/14e738b5d0cc69aa27a95dde272aea41fde44f2f/src/transformers/models/esm';
const src = (symbol: string, file: string, line: number) => ({
	symbol,
	repo: 'huggingface/transformers',
	file,
	url: `${REPO}/${file}#L${line}`
});

const S: Lane = { id: 's', label: 's  (L, 1024)', kind: 'seq' };
const Z: Lane = { id: 'z', label: 'z  (L, L, 128)', kind: 'pair' };

// Flow chart columns. Children of the block group use coordinates relative to it.
const COL = { s: 0, mid: 270, z: 540, far: 810 };
const BLOCK = { x: -30, y: 590, width: 830, height: 1230 };
const IN_BLOCK = { s: COL.s - BLOCK.x, mid: COL.mid - BLOCK.x, z: COL.z - BLOCK.x };
const AFTER = BLOCK.y + BLOCK.height + 50;
// Tensor pills are narrower than module boxes; this centres one under a module.
const T = 20;

const SM = structureModule({
	cols: COL,
	y: AFTER + 170,
	inputs: [{ from: 't_sm_in', kind: 'struct' }],
	sources: {
		structure: [src('EsmFoldStructureModule', 'modeling_esmfold.py', 1636), src('EsmFoldStructureModule.forward', 'modeling_esmfold.py', 1661)],
		init: [src('EsmFoldStructureModule.forward', 'modeling_esmfold.py', 1680)],
		ipa: [src('EsmFoldInvariantPointAttention', 'modeling_esmfold.py', 1369), src('EsmFoldInvariantPointAttention.forward', 'modeling_esmfold.py', 1409)],
		transition: [src('EsmFoldStructureModuleTransition', 'modeling_esmfold.py', 1613), src('EsmFoldStructureModuleTransitionLayer', 'modeling_esmfold.py', 1590)],
		backbone: [src('EsmFoldBackboneUpdate', 'modeling_esmfold.py', 1567), src('Rigid.compose_q_update_vec', 'openfold_utils/rigid_utils.py', 907)],
		angles: [src('EsmFoldAngleResnet', 'modeling_esmfold.py', 1304), src('EsmFoldAngleResnetBlock', 'modeling_esmfold.py', 1284)],
		atoms: [
			src('torsion_angles_to_frames', 'openfold_utils/feats.py', 160),
			src('frames_and_literature_positions_to_atom14_pos', 'openfold_utils/feats.py', 222)
		]
	}
});

export const esmfold: ModelSpec = {
	name: 'ESMFold',
	description:
		'Single-sequence structure prediction: a frozen ESM-2 language model feeds a 48-block folding trunk that keeps a per-residue state s and a per-pair state z, followed by an AlphaFold2-style structure module.',
	initialView: ['t_aa', 'esm2', 'combine', 'aa_embed', 'pair_init', 'trunk_input', 't_s_in'],

	modules: [
		{
			id: 'esm2',
			title: 'ESM-2 language model',
			subtitle: 'Frozen, 36 layers, 3B params',
			kind: 'lm',
			in: 'aa (L,)',
			out: '(L, 37, 2560)',
			summary:
				'A frozen protein language model (esm2_t36_3B_UR50D). ESMFold keeps the hidden state after the embedding layer and after each of the 36 transformer layers, so every residue gets 37 vectors of width 2560.',
			position: { x: COL.s, y: 80 },
			visual: {
				lanes: [{ id: 'h', label: 'h  (L+2, 2560)', kind: 'lm' }],
				repeat: 'x 36 layers',
				steps: [
					{ lane: 'h', label: 'Add <cls> and <eos>, embed tokens', shape: '(L+2, 2560)' },
					{ lane: 'h', label: 'LayerNorm' },
					{
						lane: 'h',
						label: 'Multi-head self-attention',
						note: '40 heads of width 64, rotary position embeddings on q and k',
						residual: true
					},
					{ lane: 'h', label: 'LayerNorm' },
					{ lane: 'h', label: 'Feed-forward', note: 'Linear 2560 -> 10240, GELU, Linear 10240 -> 2560', residual: true },
					{ lane: 'h', label: 'Stack all hidden states, drop <cls>/<eos>', shape: '(L, 37, 2560)' }
				]
			},
			math: [
				{ label: 'Embedding', tex: r`h^{(0)}_i = E[a_i]` },
				{
					label: 'Attention (per head, rotary rotation R)',
					tex: r`\alpha_{ij} = \operatorname{softmax}_j\!\left(\frac{(R_i q_i)^\top (R_j k_j)}{\sqrt{d_h}}\right), \qquad q_i = W_Q\,\mathrm{LN}(h_i),\; k_j = W_K\,\mathrm{LN}(h_j)`
				},
				{
					label: 'Layer update',
					tex: r`\begin{aligned} \tilde h^{(\ell)} &= h^{(\ell-1)} + \mathrm{Attn}\big(\mathrm{LN}(h^{(\ell-1)})\big) \\ h^{(\ell)} &= \tilde h^{(\ell)} + W_2\,\mathrm{GELU}\big(W_1\,\mathrm{LN}(\tilde h^{(\ell)})\big) \end{aligned}`
				},
				{ label: 'Output', tex: r`H_i = \big[h^{(0)}_i, h^{(1)}_i, \dots, h^{(36)}_i\big] \in \mathbb{R}^{37 \times 2560}` }
			],
			sources: [src('EsmLayer', 'modeling_esm.py', 458), src('EsmSelfAttention', 'modeling_esm.py', 313), src('EsmForProteinFolding.compute_language_model_representations', 'modeling_esmfold.py', 2181)],
			code: `class EsmLayer(nn.Module):
    """One of the 36 pre-LayerNorm transformer layers of ESM-2 (3B). Attribute names are simplified."""

    def __init__(self, dim=2560, num_heads=40, ffn_dim=10240):
        super().__init__()
        self.num_heads = num_heads
        self.attn_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, dim)

    def forward(self, h, rotary):                                    # h: (B, L+2, 2560)
        x = self.attn_norm(h)
        q, k, v = (
            proj(x).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)   # (B, 40, L+2, 64)
            for proj in (self.query, self.key, self.value)
        )
        q, k = rotary(q), rotary(k)                                  # rotary position embedding
        attn = F.scaled_dot_product_attention(q, k, v)               # (B, 40, L+2, 64)
        h = h + self.attn_out(attn.transpose(1, 2).flatten(-2))
        return h + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(h))))


@torch.no_grad()                                                     # ESM-2 is frozen
def language_model_representations(esm, tokens, cls_id, eos_id):     # tokens: (B, L) in the ESM vocabulary
    B = tokens.shape[0]
    cls = tokens.new_full((B, 1), cls_id)
    eos = tokens.new_full((B, 1), eos_id)
    tokens = torch.cat([cls, tokens, eos], dim=1)                    # (B, L+2)
    hidden = esm(tokens, output_hidden_states=True).hidden_states    # 37 tensors of (B, L+2, 2560)
    esm_s = torch.stack(hidden, dim=2)                               # (B, L+2, 37, 2560)
    return esm_s[:, 1:-1]                                            # (B, L, 37, 2560), <cls>/<eos> removed`
		},
		{
			id: 'combine',
			title: 'Layer mix + projection',
			subtitle: 'esm_s_combine, esm_s_mlp',
			kind: 'seq',
			in: '(L, 37, 2560)',
			out: '(L, 1024)',
			summary:
				'Collapses the 37 language-model layers into one vector per residue with a learned softmax weighting, then projects it to the trunk width with a two-layer MLP.',
			position: { x: COL.s, y: 250 },
			visual: {
				lanes: [{ id: 'x', label: 'per residue', kind: 'seq' }],
				steps: [
					{ lane: 'x', label: 'Softmax-weighted sum over layers', note: '37 learned scalars', shape: '(L, 2560)' },
					{ lane: 'x', label: 'LayerNorm' },
					{ lane: 'x', label: 'Linear 2560 -> 1024' },
					{ lane: 'x', label: 'ReLU' },
					{ lane: 'x', label: 'Linear 1024 -> 1024', shape: '(L, 1024)' }
				]
			},
			math: [
				{ label: 'Layer mixing', tex: r`e_i = \sum_{\ell=0}^{36} \operatorname{softmax}(w)_\ell \; h^{(\ell)}_i` },
				{ label: 'Projection', tex: r`s^{\mathrm{lm}}_i = W_2\,\mathrm{ReLU}\big(W_1\,\mathrm{LN}(e_i) + b_1\big) + b_2` }
			],
			sources: [src('EsmForProteinFolding.__init__', 'modeling_esmfold.py', 1993), src('EsmForProteinFolding.forward', 'modeling_esmfold.py', 2112)],
			code: `class LanguageModelProjection(nn.Module):
    """In the source these two attributes live directly on EsmForProteinFolding."""

    def __init__(self, esm_layers=36, esm_dim=2560, c_s=1024):
        super().__init__()
        self.esm_s_combine = nn.Parameter(torch.zeros(esm_layers + 1))   # one weight per hidden state
        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

    def forward(self, esm_s):                                        # (B, L, 37, 2560)
        weights = self.esm_s_combine.softmax(dim=0)                  # (37,)
        mixed = torch.einsum("l,bnlc->bnc", weights, esm_s)          # (B, L, 2560)
        return self.esm_s_mlp(mixed)                                 # (B, L, 1024)`
		},
		{
			id: 'aa_embed',
			title: 'Residue embedding',
			subtitle: 'Embedding(23, 1024)',
			kind: 'seq',
			in: 'aa (L,)',
			out: '(L, 1024)',
			summary:
				'A plain learned embedding of the residue identity, added to the projected language-model features. It gives the trunk direct access to which amino acid sits at each position.',
			position: { x: COL.mid, y: 250 },
			visual: {
				lanes: [{ id: 'x', label: 'per residue', kind: 'seq' }],
				steps: [
					{ lane: 'x', label: 'Token id', shape: '(L,) int' },
					{ lane: 'x', label: 'Embedding table lookup', note: '23 rows: 20 residues + pad, unknown, mask', shape: '(L, 1024)' }
				]
			},
			math: [{ tex: r`s^{\mathrm{aa}}_i = E_{\mathrm{aa}}[a_i], \qquad E_{\mathrm{aa}} \in \mathbb{R}^{23 \times 1024}` }],
			sources: [src('EsmForProteinFolding.__init__', 'modeling_esmfold.py', 2015), src('EsmForProteinFolding.forward', 'modeling_esmfold.py', 2118)],
			code: `n_tokens_embed = 20 + 3                                              # residues + pad, unknown, mask
embedding = nn.Embedding(n_tokens_embed, 1024, padding_idx=0)

s_s_0 = esm_s_mlp(mixed) + embedding(aa)                             # aa: (B, L) long -> (B, L, 1024)`
		},
		{
			id: 'pair_init',
			title: 'Pair initialisation',
			subtitle: 'Zeros + relative position',
			kind: 'pair',
			in: 'residx (L,)',
			out: '(L, L, 128)',
			summary:
				'The pairwise state starts with no sequence information at all: it is all zeros plus an embedding of how far apart two residues are along the chain. Everything else z knows has to be written in by the trunk.',
			paper:
				'This is why the paper can ask when sequence information enters z: at block 0 the pairwise state only encodes relative position.',
			position: { x: COL.z, y: 250 },
			visual: {
				lanes: [{ id: 'x', label: 'per residue pair', kind: 'pair' }],
				steps: [
					{ lane: 'x', label: 'Zeros', shape: '(L, L, 128)' },
					{ lane: 'x', label: 'Signed offset j - i', note: 'clamped to [-32, 32]', shape: '(L, L) int' },
					{ lane: 'x', label: 'Embedding table lookup', note: '66 rows: 65 offsets + padding', shape: '(L, L, 128)', residual: true }
				]
			},
			math: [
				{ label: 'Offset bin', tex: r`d_{ij} = \operatorname{clip}(r_j - r_i,\, -32,\, 32) + 33` },
				{ label: 'Initial pair state', tex: r`z^{0}_{ij} = \mathbf{0} + E_{\mathrm{pos}}[d_{ij}], \qquad E_{\mathrm{pos}} \in \mathbb{R}^{66 \times 128}` }
			],
			sources: [src('EsmFoldRelativePosition', 'modeling_esmfold.py', 1248), src('EsmForProteinFolding.forward', 'modeling_esmfold.py', 2114), src('EsmFoldingTrunk.forward', 'modeling_esmfold.py', 1887)],
			code: `class RelativePosition(nn.Module):
    def __init__(self, bins=32, c_z=128):
        super().__init__()
        self.bins = bins
        self.embedding = nn.Embedding(2 * bins + 2, c_z)             # 65 offsets, plus index 0 for padding

    def forward(self, residue_index):                                # (B, L) long
        diff = residue_index[:, None, :] - residue_index[:, :, None] # (B, L, L), entry [b, i, j] = j - i
        diff = diff.clamp(-self.bins, self.bins) + self.bins + 1     # shift into 1 .. 65
        return self.embedding(diff)                                  # (B, L, L, 128)


s_z_0 = s_s_0.new_zeros(B, L, L, 128)                                # no sequence information yet
z = s_z_0 + pairwise_positional_embedding(residx)                    # added inside the trunk`
		},
		{
			id: 'trunk_input',
			title: 'Trunk input',
			subtitle: 'Sum + recycling terms',
			kind: 'cross',
			in: '(L, 1024), (L, L, 128)',
			out: 's (L, 1024)   z (L, L, 128)',
			summary:
				'Adds the two sequence embeddings to form the initial s, and adds the recycled states from the previous pass. With recycling disabled, as in the paper, the recycled tensors are zero, so the only thing this step adds is the constant bias of the recycling LayerNorm.',
			paper: 'All experiments in the paper run a single pass (no recycling), so the 48 blocks are executed exactly once.',
			position: { x: COL.mid, y: 420 },
			visual: {
				lanes: [S, Z],
				steps: [
					{ lane: 's', label: 'Language-model features + residue embedding', shape: '(L, 1024)' },
					{ lane: 'z', label: 'Zeros + relative position', shape: '(L, L, 128)' },
					{ lane: 's', label: 'LayerNorm(previous s)', note: 'zeros on the first pass', residual: true },
					{ lane: 'z', label: 'LayerNorm(previous z)', note: 'zeros on the first pass', residual: true },
					{ lane: 'z', label: 'Embedding(binned previous distances)', note: '15 bins, bin 0 fixed to zero', residual: true }
				]
			},
			math: [
				{ label: 'Sequence state', tex: r`s_i = s^{\mathrm{lm}}_i + s^{\mathrm{aa}}_i + \mathrm{LN}_s\big(s^{\mathrm{prev}}_i\big)` },
				{
					label: 'Pair state',
					tex: r`z_{ij} = z^{0}_{ij} + \mathrm{LN}_z\big(z^{\mathrm{prev}}_{ij}\big) + E_{\mathrm{disto}}\big[\operatorname{bin}(\lVert x^{\mathrm{prev}}_i - x^{\mathrm{prev}}_j \rVert)\big]`
				},
				{ label: 'First pass', tex: r`s^{\mathrm{prev}} = \mathbf{0},\; z^{\mathrm{prev}} = \mathbf{0} \;\Rightarrow\; \mathrm{LN}(\mathbf{0}) = \beta \quad \text{(the LayerNorm bias)}` }
			],
			sources: [src('EsmFoldingTrunk', 'modeling_esmfold.py', 1834), src('EsmFoldingTrunk.forward', 'modeling_esmfold.py', 1865)],
			code: `class FoldingTrunk(nn.Module):
    def __init__(self, c_s=1024, c_z=128, num_blocks=48, recycle_bins=15):
        super().__init__()
        self.pairwise_positional_embedding = RelativePosition(bins=32, c_z=c_z)
        self.blocks = nn.ModuleList(TriangularSelfAttentionBlock() for _ in range(num_blocks))
        self.recycle_s_norm = nn.LayerNorm(c_s)
        self.recycle_z_norm = nn.LayerNorm(c_z)
        self.recycle_disto = nn.Embedding(recycle_bins, c_z)         # row 0 is set to zero at init
        self.structure_module = StructureModule()
        self.trunk2sm_s = nn.Linear(c_s, 384)
        self.trunk2sm_z = nn.Linear(c_z, 128)

    def forward(self, s_s_0, s_z_0, aa, residx, num_recycles=0):     # the paper uses num_recycles=0
        recycle_s = torch.zeros_like(s_s_0)
        recycle_z = torch.zeros_like(s_z_0)
        recycle_bins = torch.zeros(s_z_0.shape[:-1], dtype=torch.long, device=s_z_0.device)

        for _ in range(num_recycles + 1):
            # On the first pass the recycled tensors are zero, so each LayerNorm returns its bias.
            s = s_s_0 + self.recycle_s_norm(recycle_s)
            z = s_z_0 + self.recycle_z_norm(recycle_z) + self.recycle_disto(recycle_bins)
            z = z + self.pairwise_positional_embedding(residx)

            for block in self.blocks:                                # 48 blocks
                s, z = block(s, z)

            structure = self.structure_module(self.trunk2sm_s(s), self.trunk2sm_z(z), aa)
            recycle_s, recycle_z = s, z
            recycle_bins = distance_bins(structure["positions"][-1], num_bins=15)
        return s, z, structure`
		},

		// ---- Folding block (repeated 48 times) ----
		{
			id: 'block',
			title: 'Folding block',
			subtitle: 'x 48, separate weights per block',
			kind: 'cross',
			in: 's (L, 1024)   z (L, L, 128)',
			out: 's (L, 1024)   z (L, L, 128)',
			summary:
				'One block of the folding trunk. The sequence state is updated first, using the pairwise state only as an attention bias. The updated sequence state is then written into the pairwise state, which is refined by triangular updates. The two states only talk to each other through pair2seq and seq2pair.',
			paper:
				'The paper patches s or z at the output of a single block. Sequence patches work in blocks 0-7, pairwise patches from around block 25 on. Click any module inside the block for its details.',
			position: { x: BLOCK.x, y: BLOCK.y },
			size: { width: BLOCK.width, height: BLOCK.height },
			visual: {
				lanes: [S, Z],
				repeat: 'x 48 blocks',
				steps: [
					{ lane: 'all', label: 'pair2seq: z -> attention bias', note: 'read from z, written nowhere', shape: '(L, L, 32)' },
					{ lane: 's', label: 'Sequence self-attention', note: 'biased by pair2seq, gated', residual: true },
					{ lane: 's', label: 'Sequence MLP', note: '1024 -> 4096 -> 1024', residual: true },
					{ lane: 'all', label: 'seq2pair: s -> z', note: 'outer product and difference', shape: '(L, L, 128)', residual: true },
					{ lane: 'z', label: 'Triangle multiplication (outgoing)', residual: true },
					{ lane: 'z', label: 'Triangle multiplication (incoming)', residual: true },
					{ lane: 'z', label: 'Triangle attention (starting node)', residual: true },
					{ lane: 'z', label: 'Triangle attention (ending node)', residual: true },
					{ lane: 'z', label: 'Pair MLP', note: '128 -> 512 -> 128', residual: true }
				]
			},
			math: [
				{
					label: 'Sequence update',
					tex: r`\begin{aligned} s &\leftarrow s + \mathrm{SeqAttn}\big(\mathrm{LN}(s),\ \mathrm{bias} = \mathrm{pair2seq}(z)\big) \\ s &\leftarrow s + \mathrm{MLP}_s(s) \end{aligned}`
				},
				{
					label: 'Pair update',
					tex: r`\begin{aligned} z &\leftarrow z + \mathrm{seq2pair}(s) \\ z &\leftarrow z + \mathrm{TriMul}_{\mathrm{out}}(z) \\ z &\leftarrow z + \mathrm{TriMul}_{\mathrm{in}}(z) \\ z &\leftarrow z + \mathrm{TriAttn}_{\mathrm{start}}(z) \\ z &\leftarrow z + \mathrm{TriAttn}_{\mathrm{end}}(z) \\ z &\leftarrow z + \mathrm{MLP}_z(z) \end{aligned}`
				}
			],
			sources: [src('EsmFoldTriangularSelfAttentionBlock', 'modeling_esmfold.py', 1098), src('EsmFoldTriangularSelfAttentionBlock.forward', 'modeling_esmfold.py', 1133)],
			code: `class TriangularSelfAttentionBlock(nn.Module):
    def __init__(self, c_s=1024, c_z=128, sequence_head_width=32, pairwise_head_width=32):
        super().__init__()
        seq_heads = c_s // sequence_head_width                       # 32
        pair_heads = c_z // pairwise_head_width                      # 4

        self.pair_to_sequence = PairToSequence(c_z, seq_heads)
        self.layernorm_1 = nn.LayerNorm(c_s)
        self.seq_attention = SelfAttention(c_s, seq_heads, sequence_head_width)
        self.mlp_seq = ResidueMLP(c_s, 4 * c_s)

        self.sequence_to_pair = SequenceToPair(c_s, c_z // 2, c_z)
        self.tri_mul_out = TriangleMultiplicativeUpdate(c_z, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(c_z, outgoing=False)
        self.tri_att_start = TriangleAttention(c_z, pairwise_head_width, pair_heads, starting=True)
        self.tri_att_end = TriangleAttention(c_z, pairwise_head_width, pair_heads, starting=False)
        self.mlp_pair = ResidueMLP(c_z, 4 * c_z)

    def forward(self, s, z):                                         # s: (B, L, 1024), z: (B, L, L, 128)
        # Sequence update. z is only read here, as an attention bias.
        bias = self.pair_to_sequence(z)                              # (B, L, L, 32)
        s = s + self.seq_attention(self.layernorm_1(s), bias)
        s = self.mlp_seq(s)                                          # residual inside ResidueMLP

        # Pair update. It sees the s that was just updated above.
        z = z + self.sequence_to_pair(s)
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_att_start(z)
        z = z + self.tri_att_end(z)
        z = self.mlp_pair(z)                                         # residual inside ResidueMLP
        return s, z`
		},
		{
			id: 'pair2seq',
			title: 'pair2seq',
			subtitle: 'Pair state -> attention bias',
			kind: 'cross',
			in: 'z (L, L, 128)',
			out: 'bias (L, L, 32)',
			summary:
				'The only path from z to s. Each pair vector is projected to one scalar per attention head, and that scalar is added to the attention logit between the two residues. It decides who attends to whom, but carries no content of its own.',
			paper:
				'Stage 2 of the paper: the bias acts as a contact classifier (pairs with C-alpha distance under 8 A get a higher bias), which is how the geometry in z steers the sequence state.',
			parent: 'block',
			position: { x: IN_BLOCK.mid, y: 50 },
			visual: {
				lanes: [S, Z],
				steps: [
					{ lane: 'z', label: 'LayerNorm', shape: '(L, L, 128)' },
					{ lane: 'z', label: 'Linear 128 -> 32, no bias', note: 'one output per sequence-attention head' },
					{ lane: 'all', label: 'Bias handed to sequence attention', note: 'added to the logits, z itself is unchanged', shape: '(L, L, 32)' }
				]
			},
			math: [{ tex: r`b_{ij}^{h} = w_h^{\top}\,\mathrm{LN}(z_{ij}), \qquad h = 1, \dots, 32` }],
			sources: [src('EsmFoldPairToSequence', 'modeling_esmfold.py', 1061)],
			code: `class PairToSequence(nn.Module):
    def __init__(self, c_z=128, num_heads=32):
        super().__init__()
        self.layernorm = nn.LayerNorm(c_z)
        self.linear = nn.Linear(c_z, num_heads, bias=False)          # one scalar per attention head

    def forward(self, z):                                            # (B, L, L, 128)
        return self.linear(self.layernorm(z))                        # (B, L, L, 32)`
		},
		{
			id: 'seq_attn',
			title: 'Sequence self-attention',
			subtitle: '32 heads, pair-biased, gated',
			kind: 'seq',
			in: 's (L, 1024)   bias (L, L, 32)',
			out: 's (L, 1024)',
			summary:
				'Standard multi-head self-attention over residues, with two additions: the pair2seq bias is added to the logits, and the result is multiplied by a sigmoid gate computed from the query residue before the output projection.',
			parent: 'block',
			position: { x: IN_BLOCK.s, y: 220 },
			visual: {
				lanes: [S],
				steps: [
					{ lane: 's', label: 'LayerNorm', shape: '(L, 1024)' },
					{ lane: 's', label: 'Linear 1024 -> 3072, no bias', note: 'split into q, k, v for 32 heads of width 32' },
					{ lane: 's', label: 'Logits = q . k / sqrt(32) + pair2seq bias', shape: '(32, L, L)' },
					{ lane: 's', label: 'Softmax over keys, weighted sum of v', shape: '(L, 1024)' },
					{ lane: 's', label: 'Gate: sigmoid(Linear(x)) * y' },
					{ lane: 's', label: 'Output Linear 1024 -> 1024', residual: true }
				]
			},
			math: [
				{ label: 'Projections', tex: r`[q_i, k_i, v_i] = W\,\mathrm{LN}(s_i), \qquad q_i^h, k_i^h, v_i^h \in \mathbb{R}^{32}` },
				{ label: 'Attention weights', tex: r`\alpha_{ij}^{h} = \operatorname{softmax}_j\!\left(\frac{q_i^{h} \cdot k_j^{h}}{\sqrt{32}} + b_{ij}^{h}\right)` },
				{ label: 'Gated output', tex: r`y_i = \sigma\big(W_g\,\mathrm{LN}(s_i) + b_g\big) \odot \Big[\textstyle\sum_j \alpha_{ij}^{h} v_j^{h}\Big]_{h=1}^{32}` },
				{ label: 'Residual', tex: r`s_i \leftarrow s_i + W_o\,y_i + b_o` }
			],
			sources: [src('EsmFoldSelfAttention', 'modeling_esmfold.py', 939), src('EsmFoldTriangularSelfAttentionBlock.forward', 'modeling_esmfold.py', 1175)],
			code: `class SelfAttention(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=32, head_width=32):
        super().__init__()
        self.num_heads = num_heads
        self.scale = head_width**-0.5
        self.proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)  # q, k and v in one projection
        self.g_proj = nn.Linear(embed_dim, embed_dim)                # gate, initialised to weight 0, bias 1
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, bias):                                      # x: (B, L, 1024), bias: (B, L, L, 32)
        t = self.proj(x).unflatten(-1, (self.num_heads, -1))         # (B, L, 32, 96)
        q, k, v = t.transpose(1, 2).chunk(3, dim=-1)                 # each (B, 32, L, 32)

        logits = (self.scale * q) @ k.transpose(-1, -2)              # (B, 32, L, L)
        logits = logits + bias.permute(0, 3, 1, 2)                   # pair2seq bias, one map per head
        attn = logits.softmax(dim=-1)

        y = (attn @ v).transpose(1, 2).flatten(-2)                   # (B, L, 1024)
        y = torch.sigmoid(self.g_proj(x)) * y                        # gate from the query residue
        return self.o_proj(y)


# In the block:
s = s + seq_attention(layernorm_1(s), bias=pair_to_sequence(z))`
		},
		{
			id: 'seq_mlp',
			title: 'Sequence MLP',
			subtitle: '1024 -> 4096 -> 1024',
			kind: 'seq',
			in: 's (L, 1024)',
			out: 's (L, 1024)',
			summary: 'A per-residue feed-forward layer with a residual connection. Its output is the sequence state that leaves the block.',
			paper: 'The sequence representation patched and probed in the paper is s at this point, the block output.',
			parent: 'block',
			position: { x: IN_BLOCK.s, y: 320 },
			visual: {
				lanes: [S],
				steps: [
					{ lane: 's', label: 'LayerNorm' },
					{ lane: 's', label: 'Linear 1024 -> 4096' },
					{ lane: 's', label: 'ReLU' },
					{ lane: 's', label: 'Linear 4096 -> 1024', residual: true }
				]
			},
			math: [{ tex: r`s_i \leftarrow s_i + W_2\,\mathrm{ReLU}\big(W_1\,\mathrm{LN}(s_i) + b_1\big) + b_2` }],
			sources: [src('EsmFoldResidueMLP', 'modeling_esmfold.py', 1082)],
			code: `class ResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
        )

    def forward(self, x):
        return x + self.mlp(x)                                       # residual connection


mlp_seq = ResidueMLP(embed_dim=1024, inner_dim=4096)
s = mlp_seq(s)                                                       # (B, L, 1024)`
		},
		{
			id: 'seq2pair',
			title: 'seq2pair',
			subtitle: 'Sequence state -> pair state',
			kind: 'cross',
			in: 's (L, 1024)',
			out: 'z update (L, L, 128)',
			summary:
				'The only path from s to z. Each residue is projected to two 64-wide vectors. For every pair, their elementwise product and their difference are concatenated and projected to the pair width, then added to z.',
			paper:
				'Stage 1 of the paper: in the early blocks this module writes residue properties such as charge into z. Ablating it in a sliding window over early blocks is what breaks hairpin formation.',
			parent: 'block',
			position: { x: IN_BLOCK.mid, y: 490 },
			visual: {
				lanes: [S, Z],
				steps: [
					{ lane: 's', label: 'LayerNorm' },
					{ lane: 's', label: 'Linear 1024 -> 128', note: 'split into q and k, 64 each', shape: '2 x (L, 64)' },
					{ lane: 'all', label: 'Product q_j * k_i  and  difference q_j - k_i', note: 'for every pair (i, j), concatenated', shape: '(L, L, 128)' },
					{ lane: 'z', label: 'Linear 128 -> 128', residual: true }
				]
			},
			math: [
				{ label: 'Projection', tex: r`[q_i, k_i] = W\,\mathrm{LN}(s_i) + b, \qquad q_i, k_i \in \mathbb{R}^{64}` },
				{ label: 'Pair features', tex: r`u_{ij} = \big[\, q_j \odot k_i \;\Vert\; q_j - k_i \,\big] \in \mathbb{R}^{128}` },
				{ label: 'Residual', tex: r`z_{ij} \leftarrow z_{ij} + W_o\,u_{ij} + b_o` }
			],
			sources: [src('EsmFoldSequenceToPair', 'modeling_esmfold.py', 1023)],
			code: `class SequenceToPair(nn.Module):
    def __init__(self, c_s=1024, inner_dim=64, c_z=128):
        super().__init__()
        self.layernorm = nn.LayerNorm(c_s)
        self.proj = nn.Linear(c_s, 2 * inner_dim)
        self.o_proj = nn.Linear(2 * inner_dim, c_z)

    def forward(self, s):                                            # (B, L, 1024)
        q, k = self.proj(self.layernorm(s)).chunk(2, dim=-1)         # each (B, L, 64)
        q, k = q[:, None, :, :], k[:, :, None, :]                    # q varies with j, k varies with i
        pair = torch.cat([q * k, q - k], dim=-1)                     # (B, L, L, 128)
        return self.o_proj(pair)                                     # (B, L, L, 128)


# In the block:
z = z + sequence_to_pair(s)`
		},
		{
			id: 'tri_mul_out',
			title: 'Triangle multiplication',
			subtitle: 'Outgoing edges',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z (L, L, 128)',
			summary:
				'Updates the edge i-j from the two other edges of every triangle i-j-k, here the edges leaving i and j towards k. It lets the pair state enforce consistency between three residues without attention.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 660 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'LayerNorm' },
					{ lane: 'z', label: 'Two gated projections a and b', note: 'sigmoid(Linear) * Linear, 128 -> 128 each', shape: '2 x (L, L, 128)' },
					{ lane: 'z', label: 'Sum over the third residue k', note: 'a[i,k] * b[j,k], per channel', shape: '(L, L, 128)' },
					{ lane: 'z', label: 'LayerNorm, Linear 128 -> 128' },
					{ lane: 'z', label: 'Output gate sigmoid(Linear(z))', residual: true }
				]
			},
			math: [
				{ label: 'Gated projections', tex: r`a_{ij} = \sigma\big(W^a_g \hat z_{ij}\big) \odot W^a_p \hat z_{ij}, \qquad b_{ij} = \sigma\big(W^b_g \hat z_{ij}\big) \odot W^b_p \hat z_{ij}, \qquad \hat z = \mathrm{LN}(z)` },
				{ label: 'Outgoing triangle product', tex: r`x_{ij} = \sum_{k} a_{ik} \odot b_{jk}` },
				{ label: 'Gated residual', tex: r`z_{ij} \leftarrow z_{ij} + \sigma\big(W_g \hat z_{ij}\big) \odot W_z\,\mathrm{LN}(x_{ij})` }
			],
			sources: [src('EsmFoldTriangleMultiplicativeUpdate', 'modeling_esmfold.py', 549), src('EsmFoldTriangleMultiplicativeUpdate.forward', 'modeling_esmfold.py', 824)],
			code: `class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z=128, outgoing=True):
        super().__init__()
        self.outgoing = outgoing
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.linear_a_p = nn.Linear(c_z, c_z)                        # projection for edge a
        self.linear_a_g = nn.Linear(c_z, c_z)                        # its gate
        self.linear_b_p = nn.Linear(c_z, c_z)
        self.linear_b_g = nn.Linear(c_z, c_z)
        self.layer_norm_out = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, c_z)
        self.linear_g = nn.Linear(c_z, c_z)                          # output gate

    def forward(self, z):                                            # (B, L, L, 128)
        z = self.layer_norm_in(z)
        a = torch.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = torch.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)

        if self.outgoing:
            x = torch.einsum("bikc,bjkc->bijc", a, b)                # edges i->k and j->k
        else:
            x = torch.einsum("bkic,bkjc->bijc", a, b)                # edges k->i and k->j

        x = self.linear_z(self.layer_norm_out(x))
        return torch.sigmoid(self.linear_g(z)) * x                   # (B, L, L, 128)


tri_mul_out = TriangleMultiplicativeUpdate(c_z=128, outgoing=True)
z = z + tri_mul_out(z)`
		},
		{
			id: 'tri_mul_in',
			title: 'Triangle multiplication',
			subtitle: 'Incoming edges',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z (L, L, 128)',
			summary:
				'The mirror image of the outgoing update, with separate weights: edge i-j is updated from the edges arriving at i and j from every third residue k.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 760 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'LayerNorm' },
					{ lane: 'z', label: 'Two gated projections a and b', note: 'sigmoid(Linear) * Linear, 128 -> 128 each', shape: '2 x (L, L, 128)' },
					{ lane: 'z', label: 'Sum over the third residue k', note: 'a[k,i] * b[k,j], per channel', shape: '(L, L, 128)' },
					{ lane: 'z', label: 'LayerNorm, Linear 128 -> 128' },
					{ lane: 'z', label: 'Output gate sigmoid(Linear(z))', residual: true }
				]
			},
			math: [
				{ label: 'Incoming triangle product', tex: r`x_{ij} = \sum_{k} a_{ki} \odot b_{kj}` },
				{ label: 'Gated residual', tex: r`z_{ij} \leftarrow z_{ij} + \sigma\big(W_g \hat z_{ij}\big) \odot W_z\,\mathrm{LN}(x_{ij})`, note: 'a, b and the gates are defined as in the outgoing update.' }
			],
			sources: [src('EsmFoldTriangleMultiplicativeUpdate', 'modeling_esmfold.py', 549), src('EsmFoldTriangleMultiplicativeUpdate.forward', 'modeling_esmfold.py', 824)],
			code: `class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z=128, outgoing=True):
        super().__init__()
        self.outgoing = outgoing
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.linear_a_p = nn.Linear(c_z, c_z)                        # projection for edge a
        self.linear_a_g = nn.Linear(c_z, c_z)                        # its gate
        self.linear_b_p = nn.Linear(c_z, c_z)
        self.linear_b_g = nn.Linear(c_z, c_z)
        self.layer_norm_out = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, c_z)
        self.linear_g = nn.Linear(c_z, c_z)                          # output gate

    def forward(self, z):                                            # (B, L, L, 128)
        z = self.layer_norm_in(z)
        a = torch.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = torch.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)

        if self.outgoing:
            x = torch.einsum("bikc,bjkc->bijc", a, b)                # edges i->k and j->k
        else:
            x = torch.einsum("bkic,bkjc->bijc", a, b)                # edges k->i and k->j

        x = self.linear_z(self.layer_norm_out(x))
        return torch.sigmoid(self.linear_g(z)) * x                   # (B, L, L, 128)


tri_mul_in = TriangleMultiplicativeUpdate(c_z=128, outgoing=False)
z = z + tri_mul_in(z)`
		},
		{
			id: 'tri_att_start',
			title: 'Triangle attention',
			subtitle: 'Around starting node, 4 heads',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z (L, L, 128)',
			summary:
				'Self-attention along each row of z. Edge i-j attends to the other edges i-k that start at the same residue, and the logit is biased by the third edge j-k that closes the triangle.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 860 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'LayerNorm' },
					{ lane: 'z', label: 'Triangle bias: Linear 128 -> 4, no bias', note: 'from edge j-k', shape: '(4, L, L)' },
					{ lane: 'z', label: 'q, k, v projections', note: '4 heads of width 32', shape: '(L, 4, L, 32)' },
					{ lane: 'z', label: 'Row-wise attention', note: 'softmax over k for each row i' },
					{ lane: 'z', label: 'Gate and output Linear 128 -> 128', residual: true }
				]
			},
			math: [
				{ label: 'Attention along row i', tex: r`\alpha_{ijk}^{h} = \operatorname{softmax}_k\!\left(\frac{q_{ij}^{h} \cdot k_{ik}^{h}}{\sqrt{32}} + b_{jk}^{h}\right), \qquad b_{jk}^{h} = w_h^{\top} \hat z_{jk}` },
				{ label: 'Gated residual', tex: r`z_{ij} \leftarrow z_{ij} + W_o\Big( \sigma\big(W_g \hat z_{ij}\big) \odot \Big[\textstyle\sum_k \alpha_{ijk}^{h} v_{ik}^{h}\Big]_{h=1}^{4} \Big)` }
			],
			sources: [src('EsmFoldTriangleAttention', 'modeling_esmfold.py', 439), src('EsmFoldAttention', 'modeling_esmfold.py', 287)],
			code: `class TriangleAttention(nn.Module):
    def __init__(self, c_z=128, c_hidden=32, no_heads=4, starting=True):
        super().__init__()
        self.starting, self.c_hidden, self.no_heads = starting, c_hidden, no_heads
        self.layer_norm = nn.LayerNorm(c_z)
        self.linear = nn.Linear(c_z, no_heads, bias=False)           # triangle bias from the third edge
        # The five layers below sit under self.mha in the source.
        self.linear_q = nn.Linear(c_z, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c_z, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_z, c_hidden * no_heads, bias=False)
        self.linear_g = nn.Linear(c_z, c_hidden * no_heads)          # gate
        self.linear_o = nn.Linear(c_hidden * no_heads, c_z)

    def split_heads(self, t):                                        # (B, I, J, 128) -> (B, I, 4, J, 32)
        return t.unflatten(-1, (self.no_heads, self.c_hidden)).transpose(-2, -3)

    def forward(self, z):                                            # (B, L, L, 128)
        if not self.starting:
            z = z.transpose(1, 2)                                    # attend down columns instead of along rows
        x = self.layer_norm(z)

        bias = self.linear(x).permute(0, 3, 1, 2).unsqueeze(1)       # (B, 1, 4, L, L), indexed [h, j, k]
        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))

        # Row i is a sequence: edge (i, j) attends over edges (i, k).
        logits = q @ k.transpose(-1, -2) / self.c_hidden**0.5 + bias # (B, L, 4, L, L)
        out = logits.softmax(dim=-1) @ v                             # (B, L, 4, L, 32)
        out = out.transpose(-2, -3).flatten(-2)                      # (B, L, L, 128)
        out = self.linear_o(torch.sigmoid(self.linear_g(x)) * out)

        return out if self.starting else out.transpose(1, 2)


tri_att_start = TriangleAttention(c_z=128, c_hidden=32, no_heads=4, starting=True)
z = z + tri_att_start(z)`
		},
		{
			id: 'tri_att_end',
			title: 'Triangle attention',
			subtitle: 'Around ending node, 4 heads',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z (L, L, 128)',
			summary:
				'The same operation along columns: edge i-j attends to the other edges k-j that end at the same residue. It is implemented by transposing z, running the row-wise attention, and transposing back.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 960 },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Transpose (i, j) -> (j, i)' },
					{ lane: 'z', label: 'LayerNorm, triangle bias, q/k/v', note: '4 heads of width 32' },
					{ lane: 'z', label: 'Row-wise attention on the transposed state', note: 'equivalent to attention down each column' },
					{ lane: 'z', label: 'Gate and output Linear 128 -> 128' },
					{ lane: 'z', label: 'Transpose back', residual: true }
				]
			},
			math: [
				{ label: 'Attention along column j', tex: r`\alpha_{ijk}^{h} = \operatorname{softmax}_k\!\left(\frac{q_{ij}^{h} \cdot k_{kj}^{h}}{\sqrt{32}} + b_{ki}^{h}\right)` },
				{ label: 'Gated residual', tex: r`z_{ij} \leftarrow z_{ij} + W_o\Big( \sigma\big(W_g \hat z_{ij}\big) \odot \Big[\textstyle\sum_k \alpha_{ijk}^{h} v_{kj}^{h}\Big]_{h=1}^{4} \Big)` }
			],
			sources: [src('EsmFoldTriangleAttention', 'modeling_esmfold.py', 439), src('EsmFoldAttention', 'modeling_esmfold.py', 287)],
			code: `class TriangleAttention(nn.Module):
    def __init__(self, c_z=128, c_hidden=32, no_heads=4, starting=True):
        super().__init__()
        self.starting, self.c_hidden, self.no_heads = starting, c_hidden, no_heads
        self.layer_norm = nn.LayerNorm(c_z)
        self.linear = nn.Linear(c_z, no_heads, bias=False)           # triangle bias from the third edge
        # The five layers below sit under self.mha in the source.
        self.linear_q = nn.Linear(c_z, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c_z, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_z, c_hidden * no_heads, bias=False)
        self.linear_g = nn.Linear(c_z, c_hidden * no_heads)          # gate
        self.linear_o = nn.Linear(c_hidden * no_heads, c_z)

    def split_heads(self, t):                                        # (B, I, J, 128) -> (B, I, 4, J, 32)
        return t.unflatten(-1, (self.no_heads, self.c_hidden)).transpose(-2, -3)

    def forward(self, z):                                            # (B, L, L, 128)
        if not self.starting:
            z = z.transpose(1, 2)                                    # attend down columns instead of along rows
        x = self.layer_norm(z)

        bias = self.linear(x).permute(0, 3, 1, 2).unsqueeze(1)       # (B, 1, 4, L, L), indexed [h, j, k]
        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))

        # Row i is a sequence: edge (i, j) attends over edges (i, k).
        logits = q @ k.transpose(-1, -2) / self.c_hidden**0.5 + bias # (B, L, 4, L, L)
        out = logits.softmax(dim=-1) @ v                             # (B, L, 4, L, 32)
        out = out.transpose(-2, -3).flatten(-2)                      # (B, L, L, 128)
        out = self.linear_o(torch.sigmoid(self.linear_g(x)) * out)

        return out if self.starting else out.transpose(1, 2)


tri_att_end = TriangleAttention(c_z=128, c_hidden=32, no_heads=4, starting=False)
z = z + tri_att_end(z)`
		},
		{
			id: 'pair_mlp',
			title: 'Pair MLP',
			subtitle: '128 -> 512 -> 128',
			kind: 'pair',
			in: 'z (L, L, 128)',
			out: 'z (L, L, 128)',
			summary: 'A feed-forward layer applied to every pair independently, with a residual connection. Its output is the pairwise state that leaves the block.',
			paper:
				'The pairwise representation patched, probed for distance, and steered in the paper is z at this point. Linear distance probes on it reach R2 of about 0.9 in late blocks.',
			parent: 'block',
			position: { x: IN_BLOCK.z, y: 1060 },
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
			sources: [src('EsmFoldResidueMLP', 'modeling_esmfold.py', 1082)],
			code: `class ResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
        )

    def forward(self, x):
        return x + self.mlp(x)                                       # residual connection


mlp_pair = ResidueMLP(embed_dim=128, inner_dim=512)
z = mlp_pair(z)                                                      # (B, L, L, 128)`
		},

		// ---- After the trunk ----
		{
			id: 'lm_head',
			title: 'Language-model head',
			subtitle: 'Linear 1024 -> 23',
			kind: 'head',
			in: 's (L, 1024)',
			out: 'logits (L, 23)',
			summary: 'Predicts the residue identity at each position from the final sequence state. Used as an auxiliary masked-residue loss during training.',
			position: { x: COL.s, y: AFTER },
			visual: {
				lanes: [S],
				steps: [{ lane: 's', label: 'Linear 1024 -> 23', shape: '(L, 23)' }]
			},
			math: [{ tex: r`p(a_i \mid s_i) = \operatorname{softmax}\big(W s_i + b\big)` }],
			sources: [src('EsmForProteinFolding.__init__', 'modeling_esmfold.py', 2021), src('EsmForProteinFolding.forward', 'modeling_esmfold.py', 2146)],
			code: `lm_head = nn.Linear(1024, 23)                                        # 23 = residue vocabulary size

lm_logits = lm_head(s)                                               # (B, L, 23)`
		},
		{
			id: 'trunk2sm',
			title: 'Trunk to structure module',
			subtitle: 'trunk2sm_s, trunk2sm_z',
			kind: 'struct',
			in: 's (L, 1024)   z (L, L, 128)',
			out: 'single (L, 384)   pair (L, L, 128)',
			summary:
				'Two linear layers that resize the trunk states to the widths the structure module expects. This is the single place where s and z leave the trunk.',
			paper:
				'The scaling experiment multiplies z or s here, just before the structure module, to test which one the predicted geometry depends on.',
			position: { x: COL.mid, y: AFTER },
			visual: {
				lanes: [S, Z],
				steps: [
					{ lane: 's', label: 'Linear 1024 -> 384', shape: '(L, 384)' },
					{ lane: 'z', label: 'Linear 128 -> 128', shape: '(L, L, 128)' }
				]
			},
			math: [{ tex: r`s^{\mathrm{sm}}_i = W_s s_i + b_s, \qquad z^{\mathrm{sm}}_{ij} = W_z z_{ij} + b_z` }],
			sources: [src('EsmFoldingTrunk.__init__', 'modeling_esmfold.py', 1853), src('EsmFoldingTrunk.forward', 'modeling_esmfold.py', 1910)],
			code: `trunk2sm_s = nn.Linear(1024, 384)
trunk2sm_z = nn.Linear(128, 128)

single = trunk2sm_s(s)                                               # (B, L, 384)
pair = trunk2sm_z(z)                                                 # (B, L, L, 128)
structure = structure_module(single, pair, aa)`
		},
		{
			id: 'distogram',
			title: 'Distogram head',
			subtitle: 'Linear 128 -> 64, symmetrised',
			kind: 'head',
			in: 'z (L, L, 128)',
			out: 'logits (L, L, 64)',
			summary: 'Predicts a distribution over 64 distance bins for every residue pair directly from the final pairwise state, averaged with its transpose so the result is symmetric.',
			position: { x: COL.z, y: AFTER },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Linear 128 -> 64', shape: '(L, L, 64)' },
					{ lane: 'z', label: 'Average with transpose', note: 'logits[i,j] and logits[j,i]' }
				]
			},
			math: [{ tex: r`\ell_{ij} = \tfrac{1}{2}\big(W z_{ij} + W z_{ji}\big) + b` }],
			sources: [src('EsmForProteinFolding.__init__', 'modeling_esmfold.py', 2019), src('EsmForProteinFolding.forward', 'modeling_esmfold.py', 2142)],
			code: `distogram_head = nn.Linear(128, 64)                                  # 64 distance bins

logits = distogram_head(z)                                           # (B, L, L, 64)
logits = (logits + logits.transpose(1, 2)) / 2                       # make it symmetric in (i, j)`
		},
		{
			id: 'plddt',
			title: 'pLDDT head',
			subtitle: 'Per-atom confidence',
			kind: 'head',
			in: 'states (8, L, 384)',
			out: 'pLDDT (L, 37)',
			summary:
				'Predicts how accurate each atom is expected to be, as a distribution over 50 lDDT bins. It reads the single state from the structure module, not the trunk. The reported pLDDT is the expected value from the last iteration.',
			position: { x: COL.s, y: SM.bottom },
			visual: {
				lanes: [{ id: 'x', label: 'structure module state', kind: 'struct' }],
				steps: [
					{ lane: 'x', label: 'LayerNorm', shape: '(8, L, 384)' },
					{ lane: 'x', label: 'Linear 384 -> 128' },
					{ lane: 'x', label: 'Linear 128 -> 128' },
					{ lane: 'x', label: 'Linear 128 -> 1850', note: '37 atoms x 50 bins', shape: '(8, L, 37, 50)' },
					{ lane: 'x', label: 'Softmax, expected bin centre', note: 'last iteration only', shape: '(L, 37)' }
				]
			},
			math: [{ tex: r`\mathrm{pLDDT}_{i}^{a} = \sum_{b=1}^{50} c_b \; \operatorname{softmax}\big(\ell_{i}^{a}\big)_b, \qquad c_b = \tfrac{b - 0.5}{50}` }],
			sources: [src('EsmForProteinFolding.__init__', 'modeling_esmfold.py', 2024), src('EsmForProteinFolding.forward', 'modeling_esmfold.py', 2162), src('categorical_lddt', 'modeling_esmfold.py', 1221)],
			code: `lddt_bins = 50
lddt_head = nn.Sequential(                                           # no activations between the layers
    nn.LayerNorm(384),
    nn.Linear(384, 128),
    nn.Linear(128, 128),
    nn.Linear(128, 37 * lddt_bins),
)

logits = lddt_head(structure["states"])                              # (8, B, L, 1850)
logits = logits.unflatten(-1, (37, lddt_bins))                       # (8, B, L, 37, 50)

probs = logits[-1].softmax(dim=-1)                                   # last structure-module iteration
bin_centers = (torch.arange(lddt_bins) + 0.5) / lddt_bins            # (50,)
plddt = (probs * bin_centers).sum(dim=-1)                            # (B, L, 37)`
		},
		{
			id: 'ptm',
			title: 'pTM / PAE head',
			subtitle: 'Linear 128 -> 64',
			kind: 'head',
			in: 'z (L, L, 128)',
			out: 'pTM scalar   PAE (L, L)',
			summary:
				'Predicts a distribution over the aligned error between every pair of residues from the final pairwise state. The predicted TM-score and the predicted aligned error matrix are both derived from it.',
			position: { x: COL.far, y: AFTER },
			visual: {
				lanes: [Z],
				steps: [
					{ lane: 'z', label: 'Linear 128 -> 64', note: '64 error bins up to 31 A', shape: '(L, L, 64)' },
					{ lane: 'z', label: 'Expected error per pair', shape: 'PAE: (L, L)' },
					{ lane: 'z', label: 'Expected TM-score term, max over alignments', shape: 'pTM: scalar' }
				]
			},
			math: [
				{ label: 'Error distribution', tex: r`p_{ij} = \operatorname{softmax}\big(W z_{ij} + b\big) \in \mathbb{R}^{64}` },
				{ label: 'Predicted TM-score', tex: r`\mathrm{pTM} = \max_i \frac{1}{L} \sum_j \sum_b p_{ij,b}\, \frac{1}{1 + (e_b / d_0(L))^2}` }
			],
			sources: [src('EsmForProteinFolding.__init__', 'modeling_esmfold.py', 2020), src('EsmForProteinFolding.forward', 'modeling_esmfold.py', 2167), src('compute_tm', 'openfold_utils/loss.py', 73)],
			code: `ptm_head = nn.Linear(128, 64)                                        # 64 aligned-error bins

ptm_logits = ptm_head(z)                                             # (B, L, L, 64)

# Both helpers are in transformers.models.esm.openfold_utils.
ptm = compute_tm(ptm_logits, max_bin=31, no_bins=64)                 # (B,)
pae = compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=64)   # dict with (B, L, L) error`
		},
		...SM.modules
	],

	tensors: [
		{
			id: 't_aa', label: 'Amino-acid sequence', shape: '(L,) int', kind: 'input', position: { x: COL.mid + T, y: 0 },
			description:
				'The protein to fold, as one integer token per residue. This is the only input: there is no multiple sequence alignment and no template structure.',
			axes: [{ dim: 'L', meaning: 'residues in the chain' }, { dim: 'value', meaning: 'index of the amino acid, 0-20 (20 standard residues plus unknown)' }]
		},
		{
			id: 't_residx', label: 'Residue index', shape: '(L,) int', kind: 'input', position: { x: COL.z + T, y: 0 },
			description:
				'The position of each residue along the chain, 0 to L-1. It is only used to work out how far apart two residues are in sequence.',
			axes: [{ dim: 'L', meaning: 'residues in the chain' }, { dim: 'value', meaning: 'position in the chain' }]
		},
		{
			id: 't_esm', label: 'ESM-2 hidden states', shape: '(L, 37, 2560)', kind: 'lm', position: { x: COL.s + T, y: 180 },
			description:
				'Everything the language model computed for each residue: the token embedding and the output of each of its 36 layers. Different layers carry different kinds of information, so ESMFold keeps them all and learns how to weight them.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '37', meaning: 'embedding layer + 36 transformer layers' }, { dim: '2560', meaning: 'ESM-2 hidden width' }]
		},
		{
			id: 't_slm', label: 'Language-model features', shape: '(L, 1024)', kind: 'seq', position: { x: COL.s + T, y: 350 },
			description:
				'The language-model features after mixing the 37 layers and projecting to the trunk width. This carries what ESM-2 learned from evolution about each residue in its context.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '1024', meaning: 'trunk sequence width' }]
		},
		{
			id: 't_saa', label: 'Residue embedding', shape: '(L, 1024)', kind: 'seq', position: { x: COL.mid + T, y: 350 },
			description:
				'A learned vector for the identity of each residue, independent of its neighbours. It is added to the language-model features.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '1024', meaning: 'trunk sequence width' }]
		},
		{
			id: 't_z0', label: 'Initial pair state', shape: '(L, L, 128)', kind: 'pair', position: { x: COL.z + T, y: 350 },
			description:
				'The pairwise state before the trunk has run. It contains only an embedding of the sequence separation between the two residues, and nothing about which amino acids they are.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'trunk pairwise width' }]
		},
		{
			id: 't_s_in', label: 'Sequence state s', shape: '(L, 1024)', kind: 'seq', position: { x: COL.s + T, y: 520 },
			description:
				"The sequence state entering a block: one vector per residue. For the first block it is the sum of the language-model features and the residue embedding. For later blocks it is the previous block's output.",
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '1024', meaning: 'trunk sequence width' }]
		},
		{
			id: 't_z_in', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', position: { x: COL.z + T, y: 520 },
			description:
				'The pairwise state entering a block: one vector per ordered pair of residues. It is where the trunk builds up its picture of which residues are close in space.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'trunk pairwise width' }]
		},

		{
			id: 't_bias', label: 'Attention bias', shape: '(L, L, 32)', kind: 'cross', parent: 'block', position: { x: IN_BLOCK.mid + T, y: 150 },
			description:
				'One number per residue pair per attention head. It is added to the sequence attention logits, so a large value makes residue i attend to residue j. This is the only thing the sequence state ever receives from the pairwise state.',
			axes: [{ dim: 'L, L', meaning: 'query residue i, key residue j' }, { dim: '32', meaning: 'one value per sequence-attention head' }]
		},
		{
			id: 't_s_out', label: 'Sequence state s', shape: '(L, 1024)', kind: 'seq', parent: 'block', position: { x: IN_BLOCK.s + T, y: 420 },
			description:
				'The sequence state leaving the block, after attention and the MLP. This is the s that the paper patches between proteins, and what seq2pair reads from in the same block.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '1024', meaning: 'trunk sequence width' }]
		},
		{
			id: 't_dz', label: 'Pair update', shape: '(L, L, 128)', kind: 'cross', parent: 'block', position: { x: IN_BLOCK.mid + T, y: 590 },
			description:
				"What seq2pair writes into the pairwise state: for each pair, a projection of the product and difference of the two residues' vectors. It is added to z rather than replacing it.",
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'trunk pairwise width' }]
		},
		{
			id: 't_z_out', label: 'Pairwise state z', shape: '(L, L, 128)', kind: 'pair', parent: 'block', position: { x: IN_BLOCK.z + T, y: 1160 },
			description:
				'The pairwise state leaving the block, after the triangular updates and the MLP. This is the z that the paper patches, probes for distance and steers. After the last block it goes to the structure module and the output heads.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'trunk pairwise width' }]
		},

		{
			id: 't_lm', label: 'Residue logits', shape: '(L, 23)', kind: 'head', position: { x: COL.s + T, y: AFTER + 100 },
			description:
				'A score for each possible residue identity at each position, predicted from the final sequence state. Only used as an auxiliary training loss.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '23', meaning: 'residue vocabulary: 20 amino acids + pad, unknown, mask' }]
		},
		{
			id: 't_sm_in', label: 'single, pair', shape: '(L, 384), (L, L, 128)', kind: 'struct', position: { x: COL.mid + T, y: AFTER + 100 },
			description:
				"The trunk states resized for the structure module. The sequence state is narrowed to 384 and called 'single'; the pairwise state keeps its width.",
			axes: [{ dim: 'L, 384', meaning: 'single representation, per residue' }, { dim: 'L, L, 128', meaning: 'pair representation, per residue pair' }]
		},
		{
			id: 't_disto', label: 'Distogram logits', shape: '(L, L, 64)', kind: 'head', position: { x: COL.z + T, y: AFTER + 100 },
			description:
				'A predicted distribution over the distance between every pair of residues, read directly from the final pairwise state with one linear layer.',
			axes: [{ dim: 'L, L', meaning: 'pair of residues' }, { dim: '64', meaning: 'distance bins' }]
		},
		{
			id: 't_pae', label: 'pTM, aligned error', shape: 'scalar, (L, L)', kind: 'head', position: { x: COL.far + T, y: AFTER + 100 },
			description:
				'Confidence in the relative placement of residues. The aligned error is the expected position error of residue j when the prediction is aligned on residue i. pTM summarises it as a single predicted TM-score.',
			axes: [{ dim: 'scalar', meaning: 'pTM, between 0 and 1' }, { dim: 'L, L', meaning: 'predicted aligned error in angstroms' }]
		},
		{
			id: 't_plddt', label: 'pLDDT', shape: '(L, 37)', kind: 'head', position: { x: COL.s + T, y: SM.bottom + 100 },
			description:
				'Per-atom confidence in the predicted structure, on a 0-1 scale (often shown as 0-100). High values mean the local environment of that atom is expected to be correct.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '37', meaning: 'atom types' }, { dim: 'value', meaning: 'expected lDDT' }]
		},
		...SM.tensors
	],

	edges: [
		{ from: 't_aa', to: 'esm2', kind: 'input' },
		{ from: 't_aa', to: 'aa_embed', kind: 'input' },
		{ from: 't_residx', to: 'pair_init', kind: 'input' },
		{ from: 'esm2', to: 't_esm', kind: 'lm' },
		{ from: 't_esm', to: 'combine', kind: 'lm' },
		{ from: 'combine', to: 't_slm', kind: 'seq' },
		{ from: 'aa_embed', to: 't_saa', kind: 'seq' },
		{ from: 'pair_init', to: 't_z0', kind: 'pair' },
		{ from: 't_slm', to: 'trunk_input', kind: 'seq' },
		{ from: 't_saa', to: 'trunk_input', kind: 'seq' },
		{ from: 't_z0', to: 'trunk_input', kind: 'pair' },
		{ from: 'trunk_input', to: 't_s_in', kind: 'seq' },
		{ from: 'trunk_input', to: 't_z_in', kind: 'pair' },

		{ from: 't_z_in', to: 'pair2seq', kind: 'pair' },
		{ from: 'pair2seq', to: 't_bias', kind: 'cross' },
		{ from: 't_bias', to: 'seq_attn', kind: 'cross' },
		{ from: 't_s_in', to: 'seq_attn', kind: 'seq' },
		{ from: 'seq_attn', to: 'seq_mlp', kind: 'seq' },
		{ from: 'seq_mlp', to: 't_s_out', kind: 'seq' },
		{ from: 't_s_out', to: 'seq2pair', kind: 'seq' },
		{ from: 'seq2pair', to: 't_dz', kind: 'cross' },
		{ from: 't_dz', to: 'tri_mul_out', label: 'added to z', kind: 'cross' },
		{ from: 't_z_in', to: 'tri_mul_out', kind: 'pair' },
		{ from: 'tri_mul_out', to: 'tri_mul_in', kind: 'pair' },
		{ from: 'tri_mul_in', to: 'tri_att_start', kind: 'pair' },
		{ from: 'tri_att_start', to: 'tri_att_end', kind: 'pair' },
		{ from: 'tri_att_end', to: 'pair_mlp', kind: 'pair' },
		{ from: 'pair_mlp', to: 't_z_out', kind: 'pair' },

		{ from: 't_s_out', to: 'lm_head', kind: 'seq' },
		{ from: 't_s_out', to: 'trunk2sm', kind: 'seq' },
		{ from: 't_z_out', to: 'trunk2sm', kind: 'pair' },
		{ from: 't_z_out', to: 'distogram', kind: 'pair' },
		{ from: 't_z_out', to: 'ptm', kind: 'pair' },
		{ from: 'lm_head', to: 't_lm', kind: 'head' },
		{ from: 'trunk2sm', to: 't_sm_in', kind: 'struct' },
		{ from: 'distogram', to: 't_disto', kind: 'head' },
		{ from: 'ptm', to: 't_pae', kind: 'head' },
		{ from: 't_sm_s_out', to: 'plddt', kind: 'seq' },
		{ from: 'plddt', to: 't_plddt', kind: 'head' },
		...SM.edges
	]
};
