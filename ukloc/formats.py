"""Readers/writers for the Unreal Engine formats we touch: .pak (v11, read-only) and .locres (v3)."""
import io
import struct


def _read_fstring(r):
    n = struct.unpack('<i', r.read(4))[0]
    if n == 0:
        return ''
    if n < 0:
        return r.read(-n * 2)[:-2].decode('utf-16le')
    return r.read(n)[:-1].decode('utf-8', 'replace')


def _write_fstring(s):
    if s == '':
        return struct.pack('<i', 0)
    try:
        b = s.encode('ascii')
        return struct.pack('<i', len(b) + 1) + b + b'\0'
    except UnicodeEncodeError:
        b = s.encode('utf-16le')
        return struct.pack('<i', -(len(b) // 2 + 1)) + b + b'\0\0'


# ---------------------------------------------------------------- .pak (v11)

class PakReader:
    """Minimal reader for unencrypted UE5 .pak v11 files (Oodle-compressed entries supported)."""
    FOOTER_SIZE = 221  # guid16 + encrypted1 + magic4 + ver4 + off8 + size8 + hash20 + 5*32 comp names

    def __init__(self, path):
        self.f = open(path, 'rb')
        self.f.seek(0, 2)
        self.f.seek(self.f.tell() - self.FOOTER_SIZE)
        ft = self.f.read(self.FOOTER_SIZE)
        encrypted = ft[16]
        magic, ver, ioff, isz = struct.unpack_from('<IiQQ', ft, 17)
        if magic != 0x5A6F12E1 or ver != 11:
            raise ValueError(f'unsupported pak (magic={magic:#x}, version={ver})')
        if encrypted:
            raise ValueError('pak index is encrypted')
        self.compression = [ft[61 + i * 32:61 + (i + 1) * 32].split(b'\0')[0].decode() for i in range(5)]

        self.f.seek(ioff)
        r = io.BytesIO(self.f.read(isz))
        self.mount = _read_fstring(r)
        r.read(4 + 8)  # entry count, path hash seed
        if struct.unpack('<i', r.read(4))[0]:
            r.read(8 + 8 + 20)  # path hash index
        if not struct.unpack('<i', r.read(4))[0]:
            raise ValueError('pak has no full directory index')
        fdoff, fdsz = struct.unpack('<QQ', r.read(16))
        r.read(20)
        self._encoded = r.read(struct.unpack('<i', r.read(4))[0])

        self.f.seek(fdoff)
        d = io.BytesIO(self.f.read(fdsz))
        self.files = {}
        for _ in range(struct.unpack('<i', d.read(4))[0]):
            dirname = _read_fstring(d)
            for _ in range(struct.unpack('<i', d.read(4))[0]):
                name = _read_fstring(d)
                path = (self.mount + dirname + name).replace('../../../', '', 1)
                self.files[path] = struct.unpack('<i', d.read(4))[0]

    def _decode_entry(self, off):
        b = self._encoded
        v = struct.unpack_from('<I', b, off)[0]
        off += 4
        cbs = (v & 0x3f) << 11 if (v & 0x3f) != 0x3f else None
        comp = (v >> 23) & 0x3f

        def u(is32):
            nonlocal off
            fmt, n = ('<I', 4) if is32 else ('<Q', 8)
            x = struct.unpack_from(fmt, b, off)[0]
            off += n
            return x
        offset = u((v >> 31) & 1)
        usize = u((v >> 30) & 1)
        if comp:
            u((v >> 29) & 1)  # compressed size
        if cbs is None:
            cbs = u(True)
        if (v >> 22) & 1:
            raise ValueError('encrypted entries are not supported')
        return offset, usize, comp, cbs

    def read(self, path):
        offset, usize, comp, cbs = self._decode_entry(self.files[path])
        f = self.f
        f.seek(offset + 8 + 8 + 8 + 4 + 20)  # in-data entry header: offset, size, usize, comp, hash
        if not comp:
            f.read(1 + 4)
            return f.read(usize)
        import ooz  # pyooz; imported lazily so check/build work without it
        blocks = [struct.unpack('<QQ', f.read(16)) for _ in range(struct.unpack('<I', f.read(4))[0])]
        out, remaining = [], usize
        for start, end in blocks:
            f.seek(offset + start)
            n = min(cbs or remaining, remaining)
            out.append(ooz.decompress(f.read(end - start), n))
            remaining -= n
        return b''.join(out)


# ---------------------------------------------------------------- .locres (v3)

LOCRES_VERSION = 3


def read_locres(data):
    """Return [(namespace, key, text)] in file order."""
    _, entries, strings = _parse_locres(data)
    return [(ns, k, strings[i][0]) for ns, k, i, _ in entries]


def _parse_locres(data):
    r = io.BytesIO(data)
    r.read(16)
    ver = r.read(1)[0]
    if ver != LOCRES_VERSION:
        raise ValueError(f'unsupported locres version {ver}')
    strings_off = struct.unpack('<q', r.read(8))[0]
    r.read(4)  # total entry count
    entries = []
    for _ in range(struct.unpack('<I', r.read(4))[0]):
        r.read(4)  # namespace hash
        ns = _read_fstring(r)
        for _ in range(struct.unpack('<I', r.read(4))[0]):
            r.read(4)  # key hash
            key = _read_fstring(r)
            r.read(4)  # source string hash
            index_pos = r.tell()
            entries.append((ns, key, struct.unpack('<i', r.read(4))[0], index_pos))
    r.seek(strings_off)
    strings = []
    for _ in range(struct.unpack('<i', r.read(4))[0]):
        s = _read_fstring(r)
        strings.append([s, struct.unpack('<i', r.read(4))[0]])
    return strings_off, entries, strings


def patch_locres(data, translations):
    """Replace texts for {(namespace, key): text}. Hashes are kept, so the game treats the
    new text as up to date. New strings are appended to the string table so strings shared
    with untranslated keys are not affected."""
    strings_off, entries, strings = _parse_locres(data)
    positions = {(ns, k): (i, pos) for ns, k, i, pos in entries}
    unknown = [k for k in translations if k not in positions]
    if unknown:
        raise KeyError(f'{len(unknown)} unknown keys, e.g. {unknown[:5]}')
    out = bytearray(data[:strings_off])
    for key, text in translations.items():
        old, pos = positions[key]
        strings[old][1] -= 1
        struct.pack_into('<i', out, pos, len(strings))
        strings.append([text, 1])
    out += struct.pack('<i', len(strings))
    for s, refcount in strings:
        out += _write_fstring(s) + struct.pack('<i', refcount)
    return bytes(out)
