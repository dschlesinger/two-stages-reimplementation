export type Kind = 'input' | 'lm' | 'msa' | 'seq' | 'pair' | 'cross' | 'struct' | 'head';

export interface Lane {
	id: string;
	label: string;
	kind: Kind;
}

export interface Step {
	/** Lane id, or 'all' for a step that spans every lane. */
	lane: string;
	label: string;
	note?: string;
	shape?: string;
	/** Output is added back onto the lane it sits on. */
	residual?: boolean;
}

export interface Visual {
	lanes: Lane[];
	steps: Step[];
	/** Shown as a bracket around the steps, e.g. "x 36 layers". */
	repeat?: string;
}

export interface Equation {
	label?: string;
	tex: string;
	note?: string;
}

export interface Source {
	symbol: string;
	/** GitHub owner/name of the reference implementation. */
	repo: string;
	file: string;
	url: string;
}

export interface ModuleSpec {
	id: string;
	title: string;
	subtitle: string;
	kind: Kind;
	in: string;
	out: string;
	summary: string;
	/** How this module shows up in the paper's experiments. */
	paper?: string;
	visual: Visual;
	math: Equation[];
	/** PyTorch, rewritten for clarity from the reference implementation cited in `sources`. */
	code: string;
	sources: Source[];
	position: { x: number; y: number };
	parent?: string;
	size?: { width: number; height: number };
}

/** A tensor flowing between modules: drawn as a small pill. */
export interface TensorSpec {
	id: string;
	label: string;
	shape: string;
	kind: Kind;
	description: string;
	/** What each dimension of the shape indexes. */
	axes: { dim: string; meaning: string }[];
	position: { x: number; y: number };
	parent?: string;
}

export interface EdgeSpec {
	from: string;
	to: string;
	label?: string;
	kind: Kind;
}

export interface ModelSpec {
	name: string;
	description: string;
	/** Things the chart leaves out or simplifies. */
	notes?: string;
	modules: ModuleSpec[];
	tensors: TensorSpec[];
	edges: EdgeSpec[];
	/** Node ids the chart is fitted to on first load. */
	initialView: string[];
}
