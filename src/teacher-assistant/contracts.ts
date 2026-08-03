/**
 * Drill contracts — SINGLE SOURCE OF TRUTH.
 *
 * The source is speakasap/shared/contracts/drills.contracts.ts. This header is copied
 * verbatim into every vendored file, so if you are reading it anywhere else — including
 * frontend/lib/drills/contracts.ts — you are in a generated copy and your edit will be
 * overwritten by the next sync.
 *
 * Do not edit the vendored copies in services. Edit this file, then run
 *   speakasap/shared/scripts/sync-drill-contracts.sh
 * A contract change invalidates in-flight work in other tracks: announce it.
 *
 * Spec: docs/superpowers/specs/2026-07-29-drilling-assignments-design.md
 */

/** Inline drill markup: "Ich gehe [in]{in} die Schule." — [prompt]{answer}. */
export type DrillTemplate = string;

export interface DrillBlank {
  /** 0-based, matches order of appearance in the template. */
  index: number;
  /** Placeholder shown to the student. May be an empty string (suffix drills: "Ich heiß[]{e}"). */
  prompt: string;
  /** The expected answer. NEVER serialized to a student-facing response. */
  answer: string;
  /** Additional accepted answers. NEVER serialized to a student-facing response. */
  alternatives: string[];
}

export interface ParsedTemplate {
  blanks: DrillBlank[];
  /** Template with answers substituted and markup removed. Used for hashing and search. */
  plainText: string;
}

export const DRILL_BLANK_PATTERN = /\[([^\[\]]*)\]\{([^{}]*)\}/g;

export type DrillItemSource = 'BANK_GRAMMAR' | 'BANK_SEVEN' | 'AI' | 'TEACHER';
export type DrillItemStatus = 'ACTIVE' | 'RETIRED';
export type CefrLevel = 'A1' | 'A2' | 'B1' | 'B2' | 'C1' | 'C2';

export interface DrillTopicDTO {
  id: number;
  slug: string;
  title: string;
  languageCode: string;
  materialLanguage: string;
  level: CefrLevel | null;
  /** Public grammar page URL, derived from GrammarLesson.url. Null when unmapped. */
  publicUrl: string | null;
  isNew: boolean;
}

export interface DrillItemDTO {
  id: number;
  languageCode: string;
  materialLanguage: string;
  topicSlug: string | null;
  level: CefrLevel | null;
  template: DrillTemplate;
  blanks: DrillBlank[];
  hint: string | null;
  sourceType: DrillItemSource;
  courseKey: string | null;
  lessonOrder: number | null;
  unknownWords: string[];
  hash: string;
}

export interface DrillItemSearchRequest {
  languageCode: string;
  materialLanguage: string;
  topicSlugs: string[];
  level?: CefrLevel;
  courseKey?: string;
  /** Restrict course-material items to lessons at or below this order. */
  maxLessonOrder?: number;
  /** When present, items violating the 80/20 rule against this baseline are excluded. */
  vocabularyBaseline?: string[];
  limit: number;
  excludeHashes?: string[];
  /** Deterministic ordering for tests. */
  seed?: number;
}

export interface DrillItemSearchResponse {
  items: DrillItemDTO[];
  /** Number of items matching before `limit` was applied. */
  totalAvailable: number;
}

export type VocabularySource = 'THEME' | 'ITEM' | 'LESSON_BODY';

export interface VocabularyWord {
  word: string;
  lemma: string | null;
  translation: string | null;
  lessonOrder: number;
  source: VocabularySource;
}

export interface VocabularyBaseline {
  courseKey: string;
  languageCode: string;
  maxLessonOrder: number;
  words: VocabularyWord[];
  /** Lowercased, NFC-normalized surface forms + lemmas, for O(1) membership tests. */
  index: string[];
  /** False when no vocabulary has ever been built for this course — distinct from a
   *  baseline that is legitimately empty at a low lesson order. */
  hasBaseline: boolean;
}

export interface VocabularyRatioResult {
  /** Fraction of content-word tokens present in the baseline, 0..1. */
  knownRatio: number;
  /** Content words not in the baseline, deduplicated, in order of first appearance. */
  unknownWords: string[];
  /** Per-item unknown counts, keyed by item index in the input array. */
  perItemUnknownCount: number[];
  /** True when knownRatio >= 0.8 AND every perItemUnknownCount <= 2. */
  passes: boolean;
}

export const VOCABULARY_MIN_KNOWN_RATIO = 0.8;
export const VOCABULARY_MAX_NEW_WORDS_PER_SENTENCE = 2;

export type DrillSetOrigin = 'AI' | 'BANK' | 'MIXED' | 'TEACHER';
export type DrillSetReviewState = 'GENERATING' | 'VALIDATING' | 'PENDING_REVIEW' | 'APPROVED';
export type ValidationState = 'PENDING' | 'PASS' | 'WARN' | 'FAIL' | 'OVERRIDDEN';

export type ValidationIssueCode =
  | 'MARKUP_UNPARSEABLE'
  | 'BLANK_COUNT_MISMATCH'
  | 'EMPTY_ANSWER'
  | 'WRONG_SCRIPT'
  | 'RESIDUAL_MARKUP'
  | 'DUPLICATE'
  | 'VOCABULARY_RATIO'
  | 'CLOSED_LIST_MISMATCH'
  | 'OFF_TOPIC'
  | 'UNGRAMMATICAL'
  | 'WRONG_LEVEL'
  | 'UNNATURAL';

export interface ValidationIssue {
  code: ValidationIssueCode;
  message: string;
  span?: string;
}

export interface ItemValidationResult {
  /** Position of the item within the set, 0-based. */
  itemRef: number;
  state: ValidationState;
  issues: ValidationIssue[];
  suggestedFix: { template: DrillTemplate; blanks: DrillBlank[]; hint: string | null } | null;
}

export interface DrillSetItemDTO {
  id: number;
  order: number;
  item: DrillItemDTO;
  validationState: ValidationState;
  validationIssues: ValidationIssue[];
  validatedAt: string | null;
}

export interface DrillSetDTO {
  uuid: string;
  title: string;
  languageCode: string;
  materialLanguage: string;
  level: CefrLevel | null;
  topicSlugs: string[];
  courseKey: string | null;
  lessonOrder: number | null;
  origin: DrillSetOrigin;
  reviewState: DrillSetReviewState;
  createdByTeacherId: number | null;
  instructions: string | null;
  visibility: 'SHARED' | 'PRIVATE';
  knownWordRatio: number | null;
  timesAssigned: number;
  timesSelfSelected: number;
  teacherUpvotes: number;
  studentUpvotes: number;
  popularityScore: number;
  itemCount: number;
  createdAt: string;
  approvedAt: string | null;
}

/** Full set including items and answers. Teacher auth only — never returned to a student. */
export interface DrillSetDetailDTO extends DrillSetDTO {
  items: DrillSetItemDTO[];
}

export interface DrillSetListQuery {
  languageCode?: string;
  materialLanguage?: string;
  topicSlugs?: string[];
  courseKey?: string;
  lessonOrder?: number;
  /** Full-text search over item plain text. Deliberately ignores courseKey/lessonOrder. */
  q?: string;
  sort?: 'popularity' | 'recent';
  createdBy?: number;
  reviewState?: DrillSetReviewState;
  groupBy?: 'lesson';
  limit?: number;
  offset?: number;
}

export interface DrillSetListResponse {
  sets: DrillSetDTO[];
  total: number;
  /** Present only when groupBy=lesson. Key format: `${courseKey}#${lessonOrder}` or 'unassigned'. */
  groups?: Record<string, string[]>;
}

export interface GenerateDrillRequest {
  languageCode: string;
  materialLanguage: string;
  level: CefrLevel | null;
  topics: { slug: string; title: string; focus?: string }[];
  /** The teacher's verbatim free-text request. */
  instructions: string;
  count: number;
  /** Surface forms the student already knows. */
  knownVocabulary: string[];
  maxNewWordsPerSentence: number;
  /** Few-shot examples in DrillTemplate markup. */
  exampleItems: DrillTemplate[];
  /** plainText of items already chosen; the model must not repeat these. */
  avoidTexts: string[];
  correlationId: string;
}

export interface GeneratedDrillItem {
  template: DrillTemplate;
  blanks: DrillBlank[];
  hint: string | null;
  topicSlug: string;
  newWords: string[];
}

export interface GenerateDrillResponse {
  items: GeneratedDrillItem[];
  meta: { model: string; tier: string; promptTokens: number; completionTokens: number };
}

export interface ValidateDrillRequest {
  languageCode: string;
  materialLanguage: string;
  level: CefrLevel | null;
  topics: { slug: string; title: string; focus?: string }[];
  instructions: string;
  items: { itemRef: number; template: DrillTemplate; blanks: DrillBlank[]; hint: string | null }[];
  correlationId: string;
}

export interface ValidateDrillResponse {
  results: ItemValidationResult[];
  meta: { model: string; tier: string; promptTokens: number; completionTokens: number };
}

export type DrillAssignmentStatus =
  | 'GENERATING' | 'PENDING_REVIEW' | 'ASSIGNED' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED';
export type DrillAssignmentOrigin = 'TEACHER' | 'SELF';
export type GenerationPhase =
  | 'RESOLVING' | 'BANK' | 'GENERATING' | 'VALIDATING' | 'READY' | 'FAILED';

export interface GenerationProgress {
  phase: GenerationPhase;
  generated: number;
  total: number;
  etaSeconds: number | null;
  message: string;
  /** True when the job has passed its estimate without progressing. */
  stalled: boolean;
}

export interface DrillAssignmentDTO {
  uuid: string;
  setUuid: string;
  studentId: number;
  teacherId: number | null;
  origin: DrillAssignmentOrigin;
  lessonUuid: string | null;
  title: string;
  languageCode: string;
  materialLanguage: string;
  status: DrillAssignmentStatus;
  dueAt: string | null;
  resourceLinks: { topic: string; url: string }[];
  itemCount: number;
  /** Blanks the student has answered correctly, out of the total. */
  blanksCorrect: number;
  blanksTotal: number;
  generationProgress: GenerationProgress | null;
  createdAt: string;
  assignedAt: string | null;
  completedAt: string | null;
}

/** ANSWER-FREE. This is the only item shape a student-authenticated call may return. */
export interface RunnerBlankDTO {
  index: number;
  prompt: string;
  /** Character budget for the input box. Derived from the answer length, never the answer. */
  maxLength: number;
  /** Already solved in an earlier session — render as resolved text. */
  solved: boolean;
  /** The accepted text, present ONLY when solved is true (the student already typed it). */
  solvedText: string | null;
}

export interface RunnerItemDTO {
  uuid: string;
  order: number;
  /** Template with markup stripped to positional placeholders — carries no answers. */
  segments: ({ type: 'text'; value: string } | { type: 'blank'; index: number })[];
  blanks: RunnerBlankDTO[];
  hint: string | null;
}

export interface RunnerResponse {
  assignment: DrillAssignmentDTO;
  items: RunnerItemDTO[];
}

export interface CheckBlankRequest {
  itemUuid: string;
  blankIndex: number;
  value: string;
}

export interface CheckBlankResponse {
  correct: boolean;
  /**
   * The student's own typed text, trimmed, echoed back for rendering into the
   * blank. Present only when correct, `null` otherwise — it must never carry the
   * expected answer to a student who got it wrong.
   *
   * Deliberately NOT the normalized form: normalization lowercases for
   * case-insensitive languages, so echoing it back would render "Schule" as
   * "schule" for a student who typed it correctly. Servers must pass
   * `gradeBlank`'s `acceptedText` through unchanged.
   */
  acceptedText: string | null;
  attemptNo: number;
  blanksCorrect: number;
  blanksTotal: number;
  /** Server-decided. The client must not infer completion itself. */
  assignmentCompleted: boolean;
}

export type DrillErrorCode =
  | 'ASSIGNMENT_OUTSTANDING'   // 409 — self-drill blocked by pending teacher work
  | 'SET_NOT_APPROVED'         // 403 — student tried an unapproved set
  | 'SET_AHEAD_OF_STUDENT'     // 403 — lessonOrder beyond the student's progress
  | 'UNRESOLVED_VALIDATION_FAILURES' // 409 — approve attempted with open FAILs
  | 'GENERATION_IN_PROGRESS'   // 409 — mutation attempted mid-generation
  | 'IDENTITY_UNRESOLVED';     // 503 — legacy mapping lookup unavailable, fail closed

export interface DrillErrorBody {
  statusCode: number;
  code: DrillErrorCode;
  message: string;
  /** Present on ASSIGNMENT_OUTSTANDING — the assignment the student must finish first. */
  blockingAssignmentUuid?: string;
}

/** GET /api/v1/internal/drill-assignments/by-student/:studentId */
export interface InternalStudentAssignmentsResponse {
  outstanding: DrillAssignmentDTO[];
  completedRecent: DrillAssignmentDTO[];
  /** False when any assignment is ASSIGNED or IN_PROGRESS. Mirrors the §9.3 gate. */
  selfDrillingAllowed: boolean;
}

/** GET /api/v1/internal/drill-assignments/by-teacher/:teacherId */
export interface InternalTeacherAssignmentsResponse {
  awaitingReview: number;
  assigned: number;
  completedThisWeek: number;
  /** Sets blocking a student, most recent first, capped at 20. */
  reviewQueue: { setUuid: string; title: string; studentCount: number; createdAt: string }[];
}

/** GET /api/v1/internal/drill-assignments/by-lesson/:lessonUuid */
export interface InternalLessonAssignmentsResponse {
  assignments: (DrillAssignmentDTO & { studentName: string })[];
}

/** POST /internal/users/resolve-or-provision-legacy on auth-microservice. */
export interface ResolveLegacyUserRequest {
  system: 'speakasap-portal';
  legacyUserId: number;
  email: string;
  firstName?: string;
  lastName?: string;
}

export interface ResolveLegacyUserResponse {
  authUserId: string;
  /** True when this call created the mapping rather than finding it. */
  provisioned: boolean;
}

// ---------------------------------------------------------------------------
// C10 — Teacher-facing assignment creation
//
// Added after Track F found that the wizard's two write calls had no server
// behind them: education-service exposed the student runner and a teacher
// summary, but no way to create an assignment at all.
// ---------------------------------------------------------------------------

/** GET /api/v1/drill-languages on content-service. */
export interface DrillLanguageDTO {
  /** content-service's numeric Language.id, required by CreateSetInput. */
  id: number;
  /** ISO 639-1, unique. */
  code: string;
  name: string;
}

/** A student a teacher may assign to. */
export interface DrillTeacherStudentDTO {
  /** The legacy Django integer DrillAssignment.studentId is keyed on. */
  id: number;
  name: string;
  /** Groups the student belongs to, for the wizard's group picker. */
  groupUuids: string[];
}

/** GET /api/v1/drill-assignments/teacher/students */
export interface DrillTeacherRosterResponse {
  students: DrillTeacherStudentDTO[];
  groups: { uuid: string; name: string; studentIds: number[] }[];
  /**
   * Total students on the roster before `limit`/`offset`. A teacher with 656 students
   * (production, teacher 10) needs to know the picker is showing a window, not the lot.
   */
  total: number;
  /**
   * True when more students match beyond this page. Additive with `total` so a caller
   * that ignores both still gets a working — if truncated — list rather than an error.
   */
  hasMore: boolean;
}

/** Query for GET /api/v1/drill-assignments/teacher/students. All fields optional. */
export interface DrillTeacherRosterQuery {
  /** Case-insensitive match on the resolved student name. */
  search?: string;
  limit?: number;
  offset?: number;
}

/** POST /api/v1/drill-assignments/generate */
export interface GenerateAssignmentsRequest {
  studentIds: number[];
  lessonUuid?: string | null;
  languageCode: string;
  materialLanguage: string;
  level?: CefrLevel | null;
  topicSlugs: string[];
  /** The teacher's verbatim free-text request. May be empty when topics carry it. */
  instructions: string;
  count: number;
  dueAt?: string | null;
}

export interface GenerateAssignmentsResponse {
  /** One per student, in the order the ids were supplied. */
  assignmentUuids: string[];
  /** The set the pipeline will create. Known before generation starts. */
  setUuid: string;
  batchUuid: string;
}

/** POST /api/v1/drill-assignments/assign */
export interface AssignFromSetRequest {
  setUuid: string;
  studentIds: number[];
  lessonUuid?: string | null;
  dueAt?: string | null;
}

export interface AssignFromSetResponse {
  assignments: DrillAssignmentDTO[];
}
