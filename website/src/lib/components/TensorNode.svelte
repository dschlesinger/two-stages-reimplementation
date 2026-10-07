<script lang="ts">
	import { Handle, Position, type Node, type NodeProps } from '@xyflow/svelte';
	import type { TensorSpec } from '#lib/models/types.ts';
	import { kindStyle } from '#lib/models/kinds.ts';

	let { data }: NodeProps<Node<{ tensor: TensorSpec }>> = $props();
	const t = $derived(data.tensor);
	const style = $derived(kindStyle[t.kind]);
</script>

<Handle type="target" position={Position.Top} class="!border-0 !bg-transparent" />
<div
	class="flex w-[190px] cursor-pointer flex-col items-center rounded-full border border-dashed bg-space-indigo-200 px-3 py-1 leading-tight transition hover:bg-space-indigo-400 {style.border}"
>
	<span class="text-[11px] {style.text}">{t.label}</span>
	<span class="font-mono text-[11px] text-white">{t.shape}</span>
</div>
<Handle type="source" position={Position.Bottom} class="!border-0 !bg-transparent" />
