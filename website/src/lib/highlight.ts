import { createHighlighterCore, type HighlighterCore } from 'shiki/core';
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript';

let highlighter: Promise<HighlighterCore> | undefined;

export async function highlight(code: string): Promise<string> {
	highlighter ??= createHighlighterCore({
		themes: [import('shiki/themes/vitesse-dark.mjs')],
		langs: [import('shiki/langs/python.mjs')],
		engine: createJavaScriptRegexEngine()
	});
	return (await highlighter).codeToHtml(code, { lang: 'python', theme: 'vitesse-dark' });
}
