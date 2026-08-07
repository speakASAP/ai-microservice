import { generateKeyPairSync } from 'crypto';
import { JwtUtil } from './jwt.util';

const HS_SECRET = 'test-shared-secret-value';

const { privateKey, publicKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});

// A second, unrelated keypair — stands in for an attacker or a rotated-out key.
const other = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});

describe('JwtUtil RS256', () => {
  it('signs and verifies a round trip', () => {
    const token = JwtUtil.signRS256('runlayer', privateKey);
    const payload = JwtUtil.verifyRS256(token, publicKey);

    expect(payload.serviceId).toBe('runlayer');
    expect(payload.iss).toBe('ai-microservice');
    expect(payload.exp).toBeGreaterThan(payload.iat);
  });

  it('emits an RS256 header, not HS256', () => {
    const token = JwtUtil.signRS256('runlayer', privateKey);
    const header = JSON.parse(
      Buffer.from(token.split('.')[0].replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString(),
    );
    expect(header.alg).toBe('RS256');
  });

  it('rejects a token signed by a different private key', () => {
    const forged = JwtUtil.signRS256('runlayer', other.privateKey);
    expect(() => JwtUtil.verifyRS256(forged, publicKey)).toThrow('Invalid signature');
  });

  it('rejects a tampered payload', () => {
    const token = JwtUtil.signRS256('runlayer', privateKey);
    const [h, , s] = token.split('.');
    const evil = Buffer.from(
      JSON.stringify({
        serviceId: 'admin',
        iss: 'ai-microservice',
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 3600,
      }),
    )
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');

    expect(() => JwtUtil.verifyRS256(`${h}.${evil}.${s}`, publicKey)).toThrow('Invalid signature');
  });

  it('rejects an expired token', () => {
    const token = JwtUtil.signRS256('runlayer', privateKey, -10);
    expect(() => JwtUtil.verifyRS256(token, publicKey)).toThrow('Token expired');
  });

  it('rejects a foreign issuer', () => {
    const token = JwtUtil.signRS256('runlayer', privateKey);
    const [h, p, s] = token.split('.');
    const decoded = JSON.parse(
      Buffer.from(p.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString(),
    );
    decoded.iss = 'evil-service';
    const swapped = Buffer.from(JSON.stringify(decoded))
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');
    // Signature no longer matches the edited payload, so this must fail closed.
    expect(() => JwtUtil.verifyRS256(`${h}.${swapped}.${s}`, publicKey)).toThrow();
  });

  it('rejects a malformed token', () => {
    expect(() => JwtUtil.verifyRS256('not-a-token', publicKey)).toThrow('Malformed token');
  });

  /**
   * Algorithm-confusion guard: an attacker takes the PUBLIC key (which is not a
   * secret) and uses it as an HMAC key, then relabels the header as HS256. A
   * verifier that trusts the header's `alg` would accept it.
   */
  it('does not accept an HS256 token forged with the public key as HMAC secret', () => {
    const forged = JwtUtil.sign('runlayer', publicKey);
    expect(() => JwtUtil.verifyRS256(forged, publicKey)).toThrow();
  });
});

describe('JwtUtil HS256 (legacy, retained during migration)', () => {
  it('still signs and verifies a round trip', () => {
    const token = JwtUtil.sign('runlayer', HS_SECRET);
    expect(JwtUtil.verify(token, HS_SECRET).serviceId).toBe('runlayer');
  });

  it('rejects a token signed with a rotated-out secret', () => {
    const token = JwtUtil.sign('runlayer', 'old-secret');
    expect(() => JwtUtil.verify(token, HS_SECRET)).toThrow('Invalid signature');
  });

  it('does not accept an RS256 token', () => {
    const token = JwtUtil.signRS256('runlayer', privateKey);
    expect(() => JwtUtil.verify(token, HS_SECRET)).toThrow();
  });
});
