<script lang="ts">
	import { Handle, Position, type Node, type NodeProps } from '@xyflow/svelte';
	import type { ModuleSpec } from '#lib/models/types.ts';
	import { kindStyle } from '#lib/models/kinds.ts';

	let { data }: NodeProps<Node<{ module: ModuleSpec }>> = $props();
	const m = $derived(data.module);
	const style = $derived(kindStyle[m.kind]);
</script>

<Handle type="target" position={Position.Top} class="!border-0 !bg-transparent" />
<div
	class="w-[230px] cursor-pointer rounded-lg border-2 px-3 py-2 text-left shadow-lg transition hover:brightness-125 {style.border} {style.bg}"
>
	<div class="text-sm font-semibold text-white">{m.title}</div>
	<div class="text-xs {style.text}">{m.subtitle}</div>
</div>
<Handle type="source" position={Position.Bottom} class="!border-0 !bg-transparent" />
