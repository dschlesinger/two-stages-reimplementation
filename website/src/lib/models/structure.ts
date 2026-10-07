import type { EdgeSpec, Lane, ModuleSpec, Source, TensorSpec } from './types';

// The AlphaFold2 structure module, shared by ESMFold and OpenFold. Both use the same
// computation and the same sizes (c_s 384, c_z 128, 8 shared-weight iterations); only the
// class names and line numbers in the cited source differ, so those are passed in.

const r = String.raw;

type Part = 'structure' | 'init' | 'ipa' | 'transition' | 'backbone' | 'angles' | 'atoms';

interface Options {
	/** Top-left of the section. `s`, `mid` and `z` are the chart's three column x positions. */
	cols: { s: number; mid: number; z: number };
	y: number;
	sources: Record<Part, Source[]>;
	/** Edges from the model's own tensors into the structure-module input. */
	inputs: { from: string; kind: EdgeSpec['kind'] }[];
}

const SINGLE: Lane = { id: 's', label: 'single s  (L, 384)', kind: 'seq' };
const PAIR: Lane = { id: 'z', label: 'pair z  (L, L, 128)', kind: 'pair' };
const FRAMES: Lane = { id: 'T', label: 'frames T  (L,)', kind: 'struct' };

const GROUP = { width: 830, height: 620 };
const T = 20;

export function structureModule({ cols, y, sources, inputs }: Options): {
	modules: ModuleSpec[];
	tensors: TensorSpec[];
	edges: EdgeSpec[];
	/** y coordinate just below the section, for whatever the model places next. */
	bottom: number;
} {
	const groupX = cols.s - 30;
	const groupY = y + 170;
	const inGroup = { s: 30, mid: cols.mid - groupX, z: cols.z - groupX };
	const bottom = groupY + GROUP.height + 30;

	const modules: ModuleSpec[] = [
		{
			id: 'sm_init',
			title: 'Structure module input',
			subtitle: 'Normalise, project, identity frames',
			kind: 'struct',
			in: 'single (L, 384)   pair (L, L, 128)',
			out: 's (L, 384)   z (L, L, 128)   frames T (L,)',
			summary:
				'Prepares the three things the structure module iterates on. The single and pair states are normalised, and the single state gets one linear layer. Every residue is given a rigid frame, a rotation plus a translation, and all frames start as the identity at the origin. This starting point is sometimes called the "black hole" initialisation: the whole chain begins collapsed onto one point.',
			position: { x: cols.mid, y },
			visual: {
				lanes: [SINGLE, PAIR, FRAMES],
				steps: [
					{ lane: 's', label: 'LayerNorm', note: 'kept as s_initial for the angle predictor' },
					{ lane: 's', label: 'Linear 384 -> 384' },
					{ lane: 'z', label: 'LayerNorm', note: 'not changed again by the structure module' },
					{ lane: 'T', label: 'Identity rotation, zero translation', note: 'one frame per residue', shape: 'quaternion (L, 4), translation (L, 3)' }
				]
			},
			math: [
				{ label: 'States', tex: r`s^{\mathrm{initial}}_i = \mathrm{LN}(s_i), \qquad s_i \leftarrow W s^{\mathrm{initial}}_i + b, \qquad z_{ij} \leftarrow \mathrm{LN}(z_{ij})` },
				{ label: 'Frames', tex: r`T_i = (R_i, \vec t_i) = (I, \vec 0)` }
			],
			sources: sources.init,
			code: `layer_norm_s = nn.LayerNorm(384)
layer_norm_z = nn.LayerNorm(128)
linear_in = nn.Linear(384, 384)

s_initial = layer_norm_s(single)                                     # (B, L, 384)
z = layer_norm_z(pair)                                               # (B, L, L, 128)
s = linear_in(s_initial)

# One rigid frame per residue: a unit quaternion for the rotation and a translation.
quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(B, L, 4)            # identity rotation
trans = torch.zeros(B, L, 3)                                         # every residue at the origin`
		},
		{
			id: 'structure',
			title: 'Structure module iteration',
			subtitle: 'x 8, the same weights every iteration',
			kind: 'struct',
			in: 's (L, 384)   z (L, L, 128)   frames T (L,)',
			out: 's (L, 384)   frames T (L,)   atom positions (L, 14, 3)',
			summary:
				'One refinement step. Invariant point attention lets every residue look at the others, using the pair state and the current 3D arrangement of the frames. The updated single state then predicts how to move and rotate each frame and what the side-chain torsion angles are. Frames plus angles give atom coordinates. The same weights are applied eight times, so the chain unfolds from the origin step by step. Unlike the trunk, z is only read here, never updated.',
			position: { x: groupX, y: groupY },
			size: GROUP,
			visual: {
				lanes: [SINGLE, FRAMES],
				repeat: 'x 8 iterations, shared weights',
				steps: [
					{ lane: 'all', label: 'Invariant point attention', note: 'reads s, the pair state z and the frames T', residual: true },
					{ lane: 's', label: 'LayerNorm, transition, LayerNorm', note: '3 Linear layers 384 -> 384 with ReLU', residual: true },
					{ lane: 'all', label: 'Backbone update', note: 'Linear 384 -> 6, composed onto each frame' },
					{ lane: 's', label: 'Angle ResNet', note: '7 torsion angles per residue as (sin, cos)', shape: '(L, 7, 2)' },
					{ lane: 'all', label: 'Frames + torsion angles -> atom positions', shape: '(L, 14, 3)' }
				]
			},
			math: [
				{
					label: 'One iteration',
					tex: r`\begin{aligned} s &\leftarrow \mathrm{LN}\big(s + \mathrm{IPA}(s, z, T)\big) \\ s &\leftarrow \mathrm{LN}\big(s + \mathrm{MLP}(s)\big) \\ T_i &\leftarrow T_i \circ \mathrm{BackboneUpdate}(s_i) \\ \alpha_i &= \mathrm{AngleResNet}(s_i, s^{\mathrm{initial}}_i) \\ \vec x_i^{\,a} &= \mathrm{atoms}(T_i, \alpha_i, \text{residue type}_i) \end{aligned}`
				}
			],
			sources: sources.structure,
			code: `class StructureModule(nn.Module):
    def __init__(self, c_s=384, c_z=128, no_blocks=8, trans_scale_factor=10):
        super().__init__()
        self.no_blocks, self.trans_scale_factor = no_blocks, trans_scale_factor
        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_in = nn.Linear(c_s, c_s)
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.layer_norm_ipa = nn.LayerNorm(c_s)
        self.transition = StructureModuleTransition(c_s)
        self.bb_update = nn.Linear(c_s, 6)
        self.angle_resnet = AngleResnet(c_s, c_resnet=128)

    def forward(self, single, pair, aatype):                         # (B, L, 384), (B, L, L, 128), (B, L)
        s_initial = self.layer_norm_s(single)
        z = self.layer_norm_z(pair)
        s = self.linear_in(s_initial)
        quat, trans = identity_frames(s.shape[:2])                   # (B, L, 4), (B, L, 3)

        positions, states = [], []
        for _ in range(self.no_blocks):                              # 8 iterations, the same weights each time
            rot = quat_to_rot(quat)                                  # (B, L, 3, 3)
            s = self.layer_norm_ipa(s + self.ipa(s, z, rot, trans))
            s = self.transition(s)
            quat, trans = compose_update(quat, trans, self.bb_update(s))
            _, angles = self.angle_resnet(s, s_initial)              # (B, L, 7, 2)
            positions.append(frames_and_angles_to_atom14(quat, trans * self.trans_scale_factor, angles, aatype))
            states.append(s)
        return {"positions": torch.stack(positions), "states": torch.stack(states)}`
		},
		{
			id: 'ipa',
			title: 'Invariant point attention',
			subtitle: '12 heads, scalar + pair + 3D point terms',
			kind: 'cross',
			in: 's (L, 384)   z (L, L, 128)   frames T (L,)',
			out: 's update (L, 384)',
			summary:
				'Attention between residues whose logits have three parts: the usual query-key dot product, a bias from the pair state, and a distance term. For the distance term every residue predicts a few 3D points in its own local frame, which are moved into the shared global frame; residues whose points land close together attend to each other more. Because only distances between points are used, rotating or translating the whole protein leaves the result unchanged, which is where "invariant" comes from. This is the one place the pair state enters the structure module.',
			parent: 'structure',
			position: { x: inGroup.mid, y: 50 },
			visual: {
				lanes: [SINGLE, PAIR, FRAMES],
				steps: [
					{ lane: 's', label: 'Scalar q, k, v: Linear 384 -> 192', note: '12 heads of width 16' },
					{ lane: 's', label: 'Points: 4 query/key and 8 value points per head', note: 'Linear 384 -> 144 and 384 -> 432, read as (x, y, z)', shape: 'local frame' },
					{ lane: 'T', label: 'Move every point to the global frame', note: 'R * p + t' },
					{ lane: 'z', label: 'Pair bias: Linear 128 -> 12', shape: '(12, L, L)' },
					{ lane: 'all', label: 'Logits = scalar q.k  +  pair bias  -  squared distance between points', note: 'softmax over residues j', shape: '(12, L, L)' },
					{ lane: 'all', label: 'Attend over scalar values, value points and the pair state', note: 'points go back to the local frame; their norms are added as features' },
					{ lane: 's', label: 'Concatenate, Linear 2112 -> 384', residual: true }
				]
			},
			math: [
				{
					label: 'Attention weights',
					tex: r`\alpha_{ij}^{h} = \operatorname{softmax}_j\!\Big( \tfrac{1}{\sqrt{3}} \Big[ \tfrac{1}{\sqrt{c}}\, q_i^{h} \cdot k_j^{h} + b_{ij}^{h} \Big] - \tfrac{\gamma^{h} w_C}{2\sqrt{3}} \sum_{p=1}^{4} \big\lVert T_i \circ \vec q_i^{\,hp} - T_j \circ \vec k_j^{\,hp} \big\rVert^2 \Big)`,
					note: 'c = 16. gamma is a learned per-head weight passed through softplus, and w_C = sqrt(2 / (9 * 4)) balances the point term against the other two.'
				},
				{ label: 'Scalar and pair outputs', tex: r`o_i^{h} = \sum_j \alpha_{ij}^{h} v_j^{h}, \qquad \tilde o_i^{h} = \sum_j \alpha_{ij}^{h} z_{ij}` },
				{ label: 'Point output, back in the local frame', tex: r`\vec o_i^{\,hp} = T_i^{-1} \circ \sum_j \alpha_{ij}^{h} \big( T_j \circ \vec v_j^{\,hp} \big)` },
				{ label: 'Update', tex: r`s_i \leftarrow s_i + W \big[\, o_i \,\Vert\, \vec o_i \,\Vert\, \lVert \vec o_i \rVert \,\Vert\, \tilde o_i \,\big] + b` }
			],
			sources: sources.ipa,
			code: `class InvariantPointAttention(nn.Module):
    def __init__(self, c_s=384, c_z=128, c_hidden=16, no_heads=12, no_qk_points=4, no_v_points=8):
        super().__init__()
        self.c_hidden, self.no_heads = c_hidden, no_heads
        self.no_qk_points, self.no_v_points = no_qk_points, no_v_points
        self.linear_q = nn.Linear(c_s, no_heads * c_hidden)
        self.linear_kv = nn.Linear(c_s, 2 * no_heads * c_hidden)
        self.linear_q_points = nn.Linear(c_s, no_heads * no_qk_points * 3)
        self.linear_kv_points = nn.Linear(c_s, no_heads * (no_qk_points + no_v_points) * 3)
        self.linear_b = nn.Linear(c_z, no_heads)                     # pair bias
        self.head_weights = nn.Parameter(torch.zeros(no_heads))      # weight of the point term, per head
        self.linear_out = nn.Linear(no_heads * (c_z + c_hidden + no_v_points * 4), c_s)

    def global_points(self, linear, s, rot, trans):
        """Predict points in each residue's own frame, then move them to the global frame."""
        p = torch.stack(linear(s).chunk(3, dim=-1), dim=-1)          # (B, L, heads * points, 3)
        p = torch.einsum("blij,blpj->blpi", rot, p) + trans[:, :, None]
        return p.unflatten(2, (self.no_heads, -1))                   # (B, L, 12, points, 3)

    def forward(self, s, z, rot, trans):                             # rot: (B, L, 3, 3), trans: (B, L, 3)
        q = self.linear_q(s).unflatten(-1, (self.no_heads, -1))      # (B, L, 12, 16)
        k, v = self.linear_kv(s).unflatten(-1, (self.no_heads, -1)).chunk(2, dim=-1)
        q_pts = self.global_points(self.linear_q_points, s, rot, trans)       # (B, L, 12, 4, 3)
        kv_pts = self.global_points(self.linear_kv_points, s, rot, trans)     # (B, L, 12, 12, 3)
        k_pts, v_pts = kv_pts.split([self.no_qk_points, self.no_v_points], dim=3)

        # Three logit terms, each (B, 12, L, L).
        scalar = torch.einsum("bihc,bjhc->bhij", q, k) * (1 / (3 * self.c_hidden)) ** 0.5
        pair = self.linear_b(z).permute(0, 3, 1, 2) * (1 / 3) ** 0.5
        dist2 = (q_pts[:, :, None] - k_pts[:, None, :]).pow(2).sum(dim=(-1, -2))   # (B, L, L, 12)
        gamma = F.softplus(self.head_weights) * (2 / (27 * self.no_qk_points)) ** 0.5
        point = -0.5 * (gamma * dist2).permute(0, 3, 1, 2)
        a = (scalar + pair + point).softmax(dim=-1)

        o = torch.einsum("bhij,bjhc->bihc", a, v).flatten(-2)        # scalar values: (B, L, 192)
        o_pair = torch.einsum("bhij,bijc->bihc", a, z).flatten(-2)   # pair state: (B, L, 1536)
        o_pts = torch.einsum("bhij,bjhpx->bihpx", a, v_pts)          # value points, global frame
        o_pts = torch.einsum("blji,blhpj->blhpi", rot, o_pts - trans[:, :, None, None])   # back to local frame
        o_pts = o_pts.flatten(2, 3)                                  # (B, L, 96, 3)
        o_norm = (o_pts.pow(2).sum(-1) + 1e-8).sqrt()                # (B, L, 96)

        return self.linear_out(torch.cat([o, *o_pts.unbind(-1), o_norm, o_pair], dim=-1))


# In the iteration:
s = layer_norm_ipa(s + ipa(s, z, rot, trans))`
		},
		{
			id: 'sm_transition',
			title: 'Transition',
			subtitle: '3-layer MLP, LayerNorm',
			kind: 'seq',
			in: 's (L, 384)',
			out: 's (L, 384)',
			summary: 'A small per-residue MLP with a residual connection, followed by LayerNorm. Its output is the single state for this iteration, which both the backbone update and the angle predictor read.',
			parent: 'structure',
			position: { x: inGroup.s, y: 160 },
			visual: {
				lanes: [SINGLE],
				steps: [
					{ lane: 's', label: 'Linear 384 -> 384, ReLU' },
					{ lane: 's', label: 'Linear 384 -> 384, ReLU' },
					{ lane: 's', label: 'Linear 384 -> 384', residual: true },
					{ lane: 's', label: 'LayerNorm' }
				]
			},
			math: [{ tex: r`s_i \leftarrow \mathrm{LN}\Big(s_i + W_3\,\mathrm{ReLU}\big(W_2\,\mathrm{ReLU}(W_1 s_i + b_1) + b_2\big) + b_3\Big)` }],
			sources: sources.transition,
			code: `class StructureModuleTransition(nn.Module):
    def __init__(self, c=384):
        super().__init__()
        # The three linears sit under self.layers[0] in the source.
        self.linear_1 = nn.Linear(c, c)
        self.linear_2 = nn.Linear(c, c)
        self.linear_3 = nn.Linear(c, c)
        self.layer_norm = nn.LayerNorm(c)

    def forward(self, s):                                            # (B, L, 384)
        s = s + self.linear_3(F.relu(self.linear_2(F.relu(self.linear_1(s)))))
        return self.layer_norm(s)`
		},
		{
			id: 'bb_update',
			title: 'Backbone update',
			subtitle: 'Linear 384 -> 6: rotation + translation',
			kind: 'struct',
			in: 's (L, 384)   frames T (L,)',
			out: 'frames T (L,)',
			summary:
				'Predicts a small rigid motion for each residue and applies it in that residue\'s own frame. Three of the six numbers are the vector part of a quaternion (1, b, c, d), which is normalised into a rotation; the other three are a translation. Composing in the local frame means the update does not depend on how the whole protein is oriented.',
			parent: 'structure',
			position: { x: inGroup.z, y: 330 },
			visual: {
				lanes: [SINGLE, FRAMES],
				steps: [
					{ lane: 's', label: 'Linear 384 -> 6', shape: '(b, c, d), (tx, ty, tz)' },
					{ lane: 'all', label: 'Rotation: multiply the frame quaternion by (1, b, c, d), normalise' },
					{ lane: 'all', label: 'Translation: rotate (tx, ty, tz) by the current frame, add' },
					{ lane: 'T', label: 'Updated frame', shape: 'quaternion (L, 4), translation (L, 3)' }
				]
			},
			math: [
				{ label: 'Predicted update', tex: r`(b_i, c_i, d_i, \vec t_i) = W s_i + b` },
				{
					label: 'Compose onto the frame',
					tex: r`q_i \leftarrow \frac{q_i \otimes (1, b_i, c_i, d_i)}{\lVert q_i \otimes (1, b_i, c_i, d_i) \rVert}, \qquad \vec t^{\,\mathrm{frame}}_i \leftarrow \vec t^{\,\mathrm{frame}}_i + R_i\, \vec t_i`,
					note: 'R is the rotation before the update. Translations are kept in units of 10 angstroms and scaled when atoms are built.'
				}
			],
			sources: sources.backbone,
			code: `bb_update = nn.Linear(384, 6)


def quat_multiply(a, b):                                             # Hamilton product, (..., 4) each
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def compose_update(quat, trans, update):                             # (B, L, 4), (B, L, 3), (B, L, 6)
    rot = quat_to_rot(quat)                                          # rotation before the update
    bcd, t = update[..., :3], update[..., 3:]

    step = torch.cat([torch.ones_like(bcd[..., :1]), bcd], dim=-1)   # the quaternion (1, b, c, d)
    quat = F.normalize(quat_multiply(quat, step), dim=-1)
    trans = trans + torch.einsum("blij,blj->bli", rot, t)            # move in the residue's own frame
    return quat, trans


quat, trans = compose_update(quat, trans, bb_update(s))`
		},
		{
			id: 'angle_resnet',
			title: 'Angle ResNet',
			subtitle: '7 torsion angles per residue',
			kind: 'struct',
			in: 's (L, 384)   s_initial (L, 384)',
			out: 'angles (L, 7, 2)',
			summary:
				'Predicts the torsion angles that place the atoms the backbone frame does not fix: three backbone angles and up to four side-chain chi angles. Each angle is predicted as a 2D vector and normalised to a (sin, cos) pair, which avoids the wrap-around at 360 degrees. It also reads the single state from before the first iteration.',
			parent: 'structure',
			position: { x: inGroup.mid, y: 330 },
			visual: {
				lanes: [SINGLE],
				steps: [
					{ lane: 's', label: 'ReLU, Linear 384 -> 128 on s', note: 'plus the same on s_initial, with its own Linear' },
					{ lane: 's', label: '2 residual blocks', note: 'ReLU, Linear 128 -> 128, ReLU, Linear 128 -> 128', residual: true },
					{ lane: 's', label: 'ReLU, Linear 128 -> 14', shape: '(L, 7, 2)' },
					{ lane: 's', label: 'Normalise each pair to unit length', note: '(sin, cos) of each angle' }
				]
			},
			math: [
				{ label: 'Input', tex: r`a_i = W_1\,\mathrm{ReLU}(s_i) + W_2\,\mathrm{ReLU}\big(s^{\mathrm{initial}}_i\big)` },
				{ label: 'Residual block (x 2)', tex: r`a_i \leftarrow a_i + W_b\,\mathrm{ReLU}\big(W_a\,\mathrm{ReLU}(a_i)\big)` },
				{ label: 'Angles', tex: r`\tilde\alpha_i = W_o\,\mathrm{ReLU}(a_i) \in \mathbb{R}^{7 \times 2}, \qquad \alpha_i^{f} = \tilde\alpha_i^{f} / \lVert \tilde\alpha_i^{f} \rVert` }
			],
			sources: sources.angles,
			code: `class AngleResnetBlock(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.linear_1 = nn.Linear(c, c)
        self.linear_2 = nn.Linear(c, c)

    def forward(self, a):
        return a + self.linear_2(F.relu(self.linear_1(F.relu(a))))


class AngleResnet(nn.Module):
    def __init__(self, c_s=384, c_resnet=128, no_blocks=2, no_angles=7, eps=1e-8):
        super().__init__()
        self.no_angles, self.eps = no_angles, eps
        self.linear_in = nn.Linear(c_s, c_resnet)
        self.linear_initial = nn.Linear(c_s, c_resnet)
        self.layers = nn.ModuleList(AngleResnetBlock(c_resnet) for _ in range(no_blocks))
        self.linear_out = nn.Linear(c_resnet, no_angles * 2)

    def forward(self, s, s_initial):                                 # (B, L, 384) each
        a = self.linear_in(F.relu(s)) + self.linear_initial(F.relu(s_initial))
        for layer in self.layers:
            a = layer(a)
        unnormalized = self.linear_out(F.relu(a)).unflatten(-1, (self.no_angles, 2))   # (B, L, 7, 2)
        norm = unnormalized.pow(2).sum(-1, keepdim=True).clamp(min=self.eps).sqrt()
        return unnormalized, unnormalized / norm                     # the second is (sin, cos) per angle`
		},
		{
			id: 'atom_builder',
			title: 'Frames and angles to atoms',
			subtitle: 'No learned parameters',
			kind: 'struct',
			in: 'frames T (L,)   angles (L, 7, 2)   residue types (L,)',
			out: 'atom positions (L, 14, 3)',
			summary:
				'Pure geometry. Each amino acid has ideal bond lengths and angles, stored as fixed coordinates of its atoms in a handful of rigid groups. The torsion angles turn each group relative to the previous one, the backbone frame places the result in space, and the stored atom coordinates are moved through those frames. Because bond geometry comes from a table, the output always has chemically sensible local structure.',
			parent: 'structure',
			position: { x: inGroup.mid, y: 520 },
			visual: {
				lanes: [FRAMES],
				steps: [
					{ lane: 'T', label: 'Scale frame translations by 10', note: 'internal units to angstroms' },
					{ lane: 'T', label: 'Build 8 rigid-group frames per residue', note: 'backbone, then one per torsion angle, chained along the side chain', shape: '(L, 8) frames' },
					{ lane: 'T', label: 'Look up ideal atom coordinates for each residue type', note: 'each atom belongs to one rigid group' },
					{ lane: 'T', label: 'Apply each group frame to its atoms', shape: '(L, 14, 3)' }
				]
			},
			math: [
				{ label: 'Rigid-group frames', tex: r`T_i^{g} = T_i \circ T^{\mathrm{default}}_{\mathrm{aa}(i),\, g} \circ R_x\big(\alpha_i^{g}\big)`, note: 'Side-chain groups are chained: chi2 is built on the chi1 frame, and so on.' },
				{ label: 'Atom positions', tex: r`\vec x_i^{\,a} = T_i^{g(a)} \circ \vec x^{\,\mathrm{lit}}_{\mathrm{aa}(i),\, a}` }
			],
			sources: sources.atoms,
			code: `# Outline. Both helpers are table lookups plus rigid transforms, with no learned parameters.
rigids = Rigid(quat, trans).scale_translation(10)                    # internal units -> angstroms

# (B, L, 8) frames: backbone plus one per torsion angle, from fixed per-residue default frames.
all_frames = torsion_angles_to_frames(rigids, angles, aatype, default_frames)

# Ideal coordinates of every atom in its rigid group, moved through that group's frame.
positions = frames_and_literature_positions_to_atom14_pos(
    all_frames, aatype, default_frames, group_idx, atom_mask, lit_positions
)                                                                    # (B, L, 14, 3)`
		}
	];

	const tensors: TensorSpec[] = [
		{
			id: 't_sm_s', label: 'Single state s', shape: '(L, 384)', kind: 'seq', position: { x: cols.s + T, y: y + 100 },
			description: 'The per-residue state the structure module refines. Everything about where a residue should go has to be read out of this vector.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '384', meaning: 'structure-module width' }]
		},
		{
			id: 't_sm_z', label: 'Pair state z', shape: '(L, L, 128)', kind: 'pair', position: { x: cols.mid + T, y: y + 100 },
			description: 'The trunk\'s pairwise state, normalised. The structure module never updates it: invariant point attention reads it as a bias and as values, the same tensor in all eight iterations.',
			axes: [{ dim: 'L, L', meaning: 'ordered pair of residues (i, j)' }, { dim: '128', meaning: 'pairwise width' }]
		},
		{
			id: 't_frames', label: 'Frames T', shape: '(L,) rotation + translation', kind: 'struct', position: { x: cols.z + T, y: y + 100 },
			description: 'One rigid frame per residue: where its backbone is and which way it faces. All frames start as the identity at the origin, so the chain begins as a single point and is pulled apart over the iterations.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: 'rotation', meaning: 'unit quaternion (4) or 3x3 matrix' }, { dim: 'translation', meaning: '3D position, in units of 10 angstroms' }]
		},
		{
			id: 't_sm_s_out', label: 'Single state s', shape: '(L, 384)', kind: 'seq', parent: 'structure', position: { x: inGroup.s + T, y: 260 },
			description: 'The single state after this iteration\'s attention and transition. It feeds the backbone update and the angle predictor, and is the input to the next iteration. The states from the iterations are also what the pLDDT head reads.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '384', meaning: 'structure-module width' }]
		},
		{
			id: 't_frames_out', label: 'Updated frames T', shape: '(L,) rotation + translation', kind: 'struct', parent: 'structure', position: { x: inGroup.z + T, y: 430 },
			description: 'The frames after this iteration\'s rigid update. They place the backbone atoms and are the frames the next iteration\'s attention sees.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: 'rotation', meaning: 'unit quaternion (4)' }, { dim: 'translation', meaning: '3D position, in units of 10 angstroms' }]
		},
		{
			id: 't_angles', label: 'Torsion angles', shape: '(L, 7, 2)', kind: 'struct', parent: 'structure', position: { x: inGroup.mid + T, y: 430 },
			description: 'Seven torsion angles per residue, each as a (sin, cos) pair: omega, phi and psi for the backbone, and chi1 to chi4 for the side chain. Residues with shorter side chains ignore the unused chi angles.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '7', meaning: 'omega, phi, psi, chi1-chi4' }, { dim: '2', meaning: 'sin and cos of the angle' }]
		},
		{
			id: 't_pos', label: 'Atom positions', shape: '(L, 14, 3)', kind: 'struct', position: { x: cols.mid + T, y: bottom },
			description: 'The predicted structure: 3D coordinates for up to 14 heavy atoms per residue. One set is produced per iteration; the last one is the prediction.',
			axes: [{ dim: 'L', meaning: 'residues' }, { dim: '14', meaning: 'atom slots per residue (backbone + side chain)' }, { dim: '3', meaning: 'x, y, z in angstroms' }]
		}
	];

	const edges: EdgeSpec[] = [
		...inputs.map((i) => ({ from: i.from, to: 'sm_init', kind: i.kind })),
		{ from: 'sm_init', to: 't_sm_s', kind: 'seq' },
		{ from: 'sm_init', to: 't_sm_z', kind: 'pair' },
		{ from: 'sm_init', to: 't_frames', kind: 'struct' },
		{ from: 't_sm_s', to: 'ipa', kind: 'seq' },
		{ from: 't_sm_z', to: 'ipa', label: 'pair bias', kind: 'pair' },
		{ from: 't_frames', to: 'ipa', kind: 'struct' },
		{ from: 'ipa', to: 'sm_transition', kind: 'seq' },
		{ from: 'sm_transition', to: 't_sm_s_out', kind: 'seq' },
		{ from: 't_sm_s_out', to: 'bb_update', kind: 'seq' },
		{ from: 't_frames', to: 'bb_update', kind: 'struct' },
		{ from: 'bb_update', to: 't_frames_out', kind: 'struct' },
		{ from: 't_sm_s_out', to: 'angle_resnet', kind: 'seq' },
		{ from: 'angle_resnet', to: 't_angles', kind: 'struct' },
		{ from: 't_frames_out', to: 'atom_builder', kind: 'struct' },
		{ from: 't_angles', to: 'atom_builder', kind: 'struct' },
		{ from: 'atom_builder', to: 't_pos', kind: 'struct' }
	];

	return { modules, tensors, edges, bottom };
}
