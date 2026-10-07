import type { Kind } from './types';

// Full class strings so Tailwind can see them.
export const kindStyle: Record<Kind, { border: string; bg: string; text: string; dot: string; stroke: string }> = {
	input: { border: 'border-lime-cream-400', bg: 'bg-lime-cream-100', text: 'text-lime-cream-600', dot: 'bg-lime-cream-500', stroke: 'var(--color-lime-cream-400)' },
	lm: { border: 'border-space-indigo-700', bg: 'bg-space-indigo-400', text: 'text-space-indigo-800', dot: 'bg-space-indigo-700', stroke: 'var(--color-space-indigo-700)' },
	msa: { border: 'border-verdigris-700', bg: 'bg-verdigris-200', text: 'text-verdigris-800', dot: 'bg-verdigris-700', stroke: 'var(--color-verdigris-700)' },
	seq: { border: 'border-verdigris-500', bg: 'bg-verdigris-100', text: 'text-verdigris-700', dot: 'bg-verdigris-500', stroke: 'var(--color-verdigris-500)' },
	pair: { border: 'border-tiger-flame-500', bg: 'bg-tiger-flame-100', text: 'text-tiger-flame-700', dot: 'bg-tiger-flame-500', stroke: 'var(--color-tiger-flame-500)' },
	cross: { border: 'border-classic-crimson-500', bg: 'bg-classic-crimson-100', text: 'text-classic-crimson-700', dot: 'bg-classic-crimson-500', stroke: 'var(--color-classic-crimson-500)' },
	struct: { border: 'border-lime-cream-300', bg: 'bg-lime-cream-100', text: 'text-lime-cream-500', dot: 'bg-lime-cream-300', stroke: 'var(--color-lime-cream-300)' },
	head: { border: 'border-space-indigo-800', bg: 'bg-space-indigo-300', text: 'text-space-indigo-900', dot: 'bg-space-indigo-800', stroke: 'var(--color-space-indigo-800)' }
};

export const kindLabel: Record<Kind, string> = {
	input: 'Input',
	lm: 'Language model',
	msa: 'MSA state m',
	seq: 'Per-residue state s',
	pair: 'Pairwise state z',
	cross: 'Between s and z',
	struct: 'Structure',
	head: 'Output head'
};
