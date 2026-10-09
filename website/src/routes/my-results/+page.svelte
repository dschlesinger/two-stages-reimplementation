<script lang="ts">
	import modulePatching from '#lib/assets/module_patching_success.png';
</script>

<div class="max-w-4xl">
	<h1 class="text-2xl font-semibold text-white">My results</h1>

	<h2 class="mt-6 text-lg font-semibold text-white">Full patching (ESMFold)</h2>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">
		Patching the donor hairpin's sequence and pairwise representations into a helical target at all 48 trunk blocks produces
		a hairpin in 1,865 of 4,870 cases (38.3%), matching the 38.3% the paper reports for the standard intra mask. None of the
		unpatched targets contain a hairpin.
	</p>

	<figure class="mt-4">
		<img
			src={modulePatching}
			alt="Bar chart of hairpin transfer success rate by patched module. Paper: trunk with touch mask 62.5%, trunk with standard intra mask 38.3%, structure module 4.6%, input intervention 2.6%, ESM encoder 0.6%. Our reproduction: trunk with standard intra mask 38.3%, unpatched target 0.0%."
			width="1600"
			height="919"
			class="h-auto w-full rounded-md border border-space-indigo-500"
		/>
		<figcaption class="mt-2 text-xs leading-relaxed text-space-indigo-700">
			Hairpin transfer success by patched module, after Fig. 12 of the paper. The paper's values are read off its figure; the
			touch mask, structure module, input intervention and ESM encoder experiments have not been reproduced yet.
		</figcaption>
	</figure>
</div>
