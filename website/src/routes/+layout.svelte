<script lang="ts">
	import './layout.css';
	import tree from '#lib/assets/tree.png';
	import { page } from '$app/state';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	const links = [
		{ href: '/', label: 'Models' },
		{ href: '/paper-results', label: 'Paper results' },
		{ href: '/my-results', label: 'My results' }
	];
</script>

<svelte:head>
	<link rel="icon" type="image/png" href={tree} />
	<title>Two Stages of Folding</title>
</svelte:head>

<div class="flex min-h-screen flex-col">
	<header class="sticky top-0 z-10 border-b border-space-indigo-500 bg-space-indigo-300/90 backdrop-blur">
		<nav class="mx-auto flex max-w-6xl flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3 sm:px-6">
			<a href="/" class="flex items-center gap-2 font-semibold tracking-tight text-white">
				<img src={tree} alt="" class="size-8" />
				Two Stages of Folding
			</a>
			<ul class="flex items-center gap-1 text-sm">
				{#each links as { href, label } (href)}
					{@const active = page.url.pathname === href}
					<li>
						<a
							{href}
							aria-current={active ? 'page' : undefined}
							class="rounded-md px-3 py-1.5 transition-colors {active
								? 'bg-space-indigo-500 text-white'
								: 'text-space-indigo-800 hover:bg-space-indigo-400 hover:text-white'}"
						>
							{label}
						</a>
					</li>
				{/each}
			</ul>
		</nav>
	</header>

	<main class="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
		{@render children()}
	</main>
</div>
