import { createHmac, createSign, createVerify, timingSafeEqual } from 'crypto';

export interface ServiceTokenPayload {
  serviceId: string;
  iss: string;
  iat: number;
  exp: number;
}

function base64url(input: string | Buffer): string {
  const buf = typeof input === 'string' ? Buffer.from(input) : input;
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function base64urlDecode(input: string): Buffer {
  const padded = input.replace(/-/g, '+').replace(/_/g, '/');
  return Buffer.from(padded, 'base64');
}

export class JwtUtil {
  private static readonly ISSUER = 'ai-microservice';
  private static readonly ALGORITHM = 'HS256';
  private static readonly RS_ALGORITHM = 'RS256';

  static sign(serviceId: string, secret: string, expiresInSeconds = 365 * 24 * 3600): string {
    const header = base64url(JSON.stringify({ alg: this.ALGORITHM, typ: 'JWT' }));
    const now = Math.floor(Date.now() / 1000);
    const payload = base64url(
      JSON.stringify({ serviceId, iss: this.ISSUER, iat: now, exp: now + expiresInSeconds }),
    );
    const signature = base64url(
      createHmac('sha256', secret).update(`${header}.${payload}`).digest(),
    );
    return `${header}.${payload}.${signature}`;
  }

  static verify(token: string, secret: string): ServiceTokenPayload {
    const parts = token.split('.');
    if (parts.length !== 3) throw new Error('Malformed token');

    const [header, payload, signature] = parts;

    // Reject anything not labelled HS256 so an RS256 token can never be routed
    // into the HMAC path during the migration window.
    let decodedHeader: { alg?: string };
    try {
      decodedHeader = JSON.parse(base64urlDecode(header).toString()) as { alg?: string };
    } catch {
      throw new Error('Malformed token');
    }
    if (decodedHeader.alg !== this.ALGORITHM) {
      throw new Error('Unexpected token algorithm');
    }

    const expectedSig = base64url(
      createHmac('sha256', secret).update(`${header}.${payload}`).digest(),
    );

    // Constant-time comparison to prevent timing attacks
    const sigBuf = Buffer.from(signature);
    const expBuf = Buffer.from(expectedSig);
    if (sigBuf.length !== expBuf.length || !timingSafeEqual(sigBuf, expBuf)) {
      throw new Error('Invalid signature');
    }

    const decoded = JSON.parse(base64urlDecode(payload).toString()) as ServiceTokenPayload;
    const now = Math.floor(Date.now() / 1000);

    if (decoded.exp < now) throw new Error('Token expired');
    if (decoded.iss !== this.ISSUER) throw new Error('Invalid issuer');

    return decoded;
  }

  /**
   * Signs a service token with an RSA private key. Only ai-microservice holds
   * the private key, so a leaked public key cannot be used to mint tokens —
   * unlike HS256, where the verifying secret is also the signing secret.
   */
  static signRS256(
    serviceId: string,
    privateKey: string,
    expiresInSeconds = 365 * 24 * 3600,
  ): string {
    const header = base64url(JSON.stringify({ alg: this.RS_ALGORITHM, typ: 'JWT' }));
    const now = Math.floor(Date.now() / 1000);
    const payload = base64url(
      JSON.stringify({ serviceId, iss: this.ISSUER, iat: now, exp: now + expiresInSeconds }),
    );
    const signature = base64url(
      createSign('RSA-SHA256').update(`${header}.${payload}`).sign(privateKey),
    );
    return `${header}.${payload}.${signature}`;
  }

  /** Verifies an RS256 token against the public key. */
  static verifyRS256(token: string, publicKey: string): ServiceTokenPayload {
    const parts = token.split('.');
    if (parts.length !== 3) throw new Error('Malformed token');

    const [header, payload, signature] = parts;

    // Pin the algorithm from the header before verifying. Without this an
    // attacker could relabel a token as HS256 and sign it with the public key
    // (which is not secret) — the classic algorithm-confusion attack.
    let decodedHeader: { alg?: string };
    try {
      decodedHeader = JSON.parse(base64urlDecode(header).toString()) as { alg?: string };
    } catch {
      throw new Error('Malformed token');
    }
    if (decodedHeader.alg !== this.RS_ALGORITHM) {
      throw new Error('Unexpected token algorithm');
    }

    const ok = createVerify('RSA-SHA256')
      .update(`${header}.${payload}`)
      .verify(publicKey, base64urlDecode(signature));
    if (!ok) throw new Error('Invalid signature');

    const decoded = JSON.parse(base64urlDecode(payload).toString()) as ServiceTokenPayload;
    const now = Math.floor(Date.now() / 1000);

    if (decoded.exp < now) throw new Error('Token expired');
    if (decoded.iss !== this.ISSUER) throw new Error('Invalid issuer');

    return decoded;
  }
}
