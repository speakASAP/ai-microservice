import type { AnalyzeErrorsRequest, AnalyzeFailure } from './contracts';

const LANGUAGE_NAMES: Record<string, string> = {
  ru: 'Russian',
  en: 'English',
  de: 'German',
  es: 'Spanish',
};

export const ANALYZE_SYSTEM_PROMPT = `You are an experienced language teacher reviewing one student's completed exercise.

Your job is to explain the grammar behind the mistakes so the student understands the rule, not just the correction. You are writing for the student, not for the teacher.

Rules you must follow:
- Group the mistakes into grammar gaps. Each gap is one cluster.
- Use ONLY the topic slugs given in the request. Never invent a slug. If a mistake fits none of them, use the slug ending in ".other".
- Every submitted answer must appear in exactly one cluster. Never drop one, never place one in two clusters.
- Copy each answer into "answers" EXACTLY as it appears after "Correct answer:", character for character, including capitalization. German nouns and sentence-initial articles are capitalized there ("Das", "Die"); writing them lowercase because that is how they read in a sentence makes the answer unmatchable.
- Address what the student ACTUALLY TYPED. "across" written where "through" belongs is a different lesson from an empty answer.
- Explain the rule, why the student's attempt broke it, and how to choose correctly next time.
- Give two or three short example sentences per cluster, using vocabulary at or below the student's level.
- Keep the explanation to a few short paragraphs. A student reads this after finishing an exercise, not before a exam.

Return JSON only, matching the requested schema.`;

export function buildAnalyzeUserPrompt(req: AnalyzeErrorsRequest): string {
  const target = LANGUAGE_NAMES[req.languageCode] ?? req.languageCode;
  const material = LANGUAGE_NAMES[req.materialLanguage] ?? req.materialLanguage;

  const lines: string[] = [
    `The student is learning ${target}.`,
    `Level: ${req.level ?? 'unknown'}.`,
    ``,
    `Write "title", "explanation", "rules" and every example "gloss" in ${material}.`,
    `Write every example "text" in ${target}.`,
    ``,
    `Allowed topic slugs — use these and nothing else:`,
    ...req.allowedTopicSlugs.map((slug) => `- ${slug}`),
    ``,
    `The student got these blanks wrong:`,
    ``,
  ];

  req.failures.forEach((failure, index) => {
    lines.push(...describeFailure(failure, index + 1));
    lines.push('');
  });

  lines.push(
    `Group these into grammar gaps and explain each one. Every one of the ${req.failures.length} answers above must appear in exactly one cluster's "answers" array.`,
  );

  return lines.join('\n');
}

function describeFailure(failure: AnalyzeFailure, position: number): string[] {
  const lines = [
    `${position}. Sentence: ${failure.sentence}`,
    `   Correct answer: ${failure.answer}`,
  ];

  if (failure.prompt) {
    lines.push(`   Prompt shown to the student: ${failure.prompt}`);
  }

  if (failure.wrongAttempts.length > 0) {
    lines.push(`   The student typed: ${failure.wrongAttempts.join(', ')}`);
  } else {
    lines.push(`   The student typed nothing.`);
  }

  if (failure.revealed) {
    lines.push(`   The student revealed the answer — they did not know it.`);
  }

  lines.push(`   Wrong ${failure.mistakeCount} time(s).`);

  return lines;
}
