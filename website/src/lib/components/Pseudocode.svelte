<script lang="ts">
	import { highlight } from '#lib/highlight.ts';

	let { code }: { code: string } = $props();
	let html = $state('');

	$effect(() => {
		const current = code;
		html = '';
		highlight(current).then((result) => {
			if (current === code) html = result;
		});
	});
</script>

<div class="code overflow-x-auto rounded-lg bg-space-indigo-100 p-4 text-[13px] leading-relaxed">
	{#if html}
		{@html html}
	{:else}
		<pre class="font-mono text-space-indigo-900">{code}</pre>
	{/if}
</div>

<style>
	.code :global(pre) {
		background: transparent !important;
		margin: 0;
	}
</style>
