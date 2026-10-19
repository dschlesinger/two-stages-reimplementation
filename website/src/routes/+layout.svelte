<script lang="ts">
	import './layout.css';
	import tree from '#lib/assets/tree.png';
	import { page } from '$app/state';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	const links = [
		{ href: '/', label: 'Models' },
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
			<a
				href="https://github.com/dschlesinger/two-stages-reimplementation"
				target="_blank"
				rel="noopener noreferrer"
				class="ml-auto flex items-center gap-2 rounded-md px-3 py-1.5 text-sm text-space-indigo-800 transition-colors hover:bg-space-indigo-400 hover:text-white"
			>
				<svg viewBox="0 0 16 16" aria-hidden="true" class="size-4 fill-current">
					<path
						d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
					/>
				</svg>
				GitHub
			</a>
		</nav>
	</header>

	<main class="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
		{@render children()}
	</main>
</div>
