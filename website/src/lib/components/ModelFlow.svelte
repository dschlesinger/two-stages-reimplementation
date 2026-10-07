<script lang="ts">
	import { untrack } from 'svelte';
	import { SvelteFlow, Background, Controls, type Edge, type Node } from '@xyflow/svelte';
	import '@xyflow/svelte/dist/style.css';
	import type { ModelSpec, ModuleSpec, TensorSpec } from '#lib/models/types.ts';
	import { kindStyle } from '#lib/models/kinds.ts';
	import ModuleNode from './ModuleNode.svelte';
	import GroupNode from './GroupNode.svelte';
	import TensorNode from './TensorNode.svelte';
	import ModuleModal from './ModuleModal.svelte';
	import TensorModal from './TensorModal.svelte';

	let { model }: { model: ModelSpec } = $props();

	const nodeTypes = { module: ModuleNode, block: GroupNode, tensor: TensorNode };

	// The spec is static, so the chart is built once. Parents must come before their children.
	const spec = untrack(() => model);
	let nodes = $state.raw<Node[]>(
		[...spec.modules]
			.sort((a, b) => Number(!!a.parent) - Number(!!b.parent))
			.map((m): Node => ({
				id: m.id,
				type: m.size ? 'block' : 'module',
				position: m.position,
				parentId: m.parent,
				data: { module: m },
				...(m.size ? { width: m.size.width, height: m.size.height } : {})
			}))
			.concat(
				spec.tensors.map((t) => ({
					id: t.id,
					type: 'tensor',
					position: t.position,
					parentId: t.parent,
					data: { tensor: t }
				}))
			)
	);
	let edges = $state.raw<Edge[]>(
		spec.edges.map((e) => ({
			id: `${e.from}-${e.to}`,
			source: e.from,
			target: e.to,
			label: e.label,
			type: 'smoothstep',
			style: `stroke: ${kindStyle[e.kind].stroke}; stroke-width: 2;`
		}))
	);

	let selected = $state<ModuleSpec | null>(null);
	let selectedTensor = $state<TensorSpec | null>(null);

	function open(id: string) {
		selected = spec.modules.find((m) => m.id === id) ?? null;
		selectedTensor = spec.tensors.find((t) => t.id === id) ?? null;
	}
</script>

<div class="flow h-[78vh] overflow-hidden rounded-xl border border-space-indigo-500">
	<SvelteFlow
		bind:nodes
		bind:edges
		{nodeTypes}
		colorMode="dark"
		fitView
		fitViewOptions={{ nodes: model.initialView.map((id) => ({ id })), padding: 0.1 }}
		minZoom={0.1}
		panOnScroll
		nodesDraggable={false}
		nodesConnectable={false}
		elementsSelectable={false}
		proOptions={{ hideAttribution: false }}
		onnodeclick={({ node }) => open(node.id)}
	>
		<Background />
		<Controls showLock={false} />
	</SvelteFlow>
</div>

<ModuleModal bind:module={selected} />
<TensorModal bind:tensor={selectedTensor} {model} onnavigate={open} />

<style>
	.flow :global(.svelte-flow) {
		--xy-background-color: var(--color-space-indigo-200);
		--xy-edge-label-background-color: var(--color-space-indigo-200);
		--xy-edge-label-color: var(--color-space-indigo-900);
	}
	.flow :global(.svelte-flow__node) {
		cursor: pointer;
	}
</style>
