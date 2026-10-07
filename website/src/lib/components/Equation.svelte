<script lang="ts">
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	let { tex }: { tex: string } = $props();
	const html = $derived(katex.renderToString(tex, { displayMode: true, throwOnError: false }));
</script>

<div class="overflow-x-auto py-1 text-white">{@html html}</div>
