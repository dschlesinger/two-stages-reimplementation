<script lang="ts">
	import type { Node, NodeProps } from '@xyflow/svelte';
	import type { ModuleSpec } from '#lib/models/types.ts';

	let { data }: NodeProps<Node<{ module: ModuleSpec }>> = $props();
	const m = $derived(data.module);
</script>

<div
	class="h-full w-full cursor-pointer rounded-xl border-2 border-dashed border-classic-crimson-400 bg-space-indigo-300/60 transition hover:bg-space-indigo-300"
>
	<div class="flex items-baseline gap-2 px-4 py-2">
		<span class="text-sm font-semibold text-white">{m.title}</span>
		<span class="text-xs text-classic-crimson-700">{m.subtitle}</span>
	</div>
</div>
