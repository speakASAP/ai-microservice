/**
 * ai-microservice role vocabulary.
 *
 * Every machine route must carry exactly one of these constants, or @Public.
 * Classified by effect, not HTTP verb.
 *
 *   INVOKE   - inference and document extraction spend (complete, voice, shop,
 *              email-triage, teacher-assistant, task draft, documents).
 *   OPERATOR - Claude Code / Codex job enqueue (can run tools against repos).
 *
 * `internal:ai-microservice:invoke` is the role minted for per-pair service
 * JWTs from consumers. It deliberately cannot enqueue code-execution jobs.
 *
 * Service tokens never receive global:superadmin; that claim is listed only so
 * a human admin JWT that happens to hit a machine route is not blocked mid-
 * migration of the admin surface.
 */

export const AI_INVOKE_ROLE = 'internal:ai-microservice:invoke';
export const AI_OPERATOR_ROLE = 'internal:ai-microservice:operator';
export const AI_ADMIN_ROLE = 'internal:ai-microservice:admin';

export const AI_INVOKE_ROLES = [
  'global:superadmin',
  AI_ADMIN_ROLE,
  AI_OPERATOR_ROLE,
  AI_INVOKE_ROLE,
] as const;

export const AI_OPERATOR_ROLES = [
  'global:superadmin',
  AI_ADMIN_ROLE,
  AI_OPERATOR_ROLE,
] as const;

/** Roles the legacy ai-issued JWT path may satisfy during the dual window.
 *  Matches pre-migration authority (any AI_SERVICE_TOKEN could call every machine
 *  route, including claude-code). Auth-minted principals get least privilege;
 *  close ALLOW_LEGACY_AI_ISSUED only after runlayer holds operator and others invoke. */
export const AI_LEGACY_EFFECTIVE_ROLES = [AI_INVOKE_ROLE, AI_OPERATOR_ROLE] as const;
