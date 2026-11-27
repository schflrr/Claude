#!/usr/bin/env python3
"""TikTok-style narrator caption (owner-approved 2026-11-27, from the Brendan Ruh reference).

Usage: python3 -I tools/caption_box.py "<text>" <out.png> [--width 900] [--size 64]

White rounded box per line, black sentence-case Montserrat SemiBold, emoji in full colour
(Google Noto Color Emoji, Apache-2.0; Apple's emoji art is not licensed for this). Writes a
transparent 1080-wide PNG for render_clip's "overlay" (visible for the whole clip, freezes included).
"""
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "assets", "fonts", "montserrat-latin-600-normal.woff")
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMO_RE = re.compile("([\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]+)")


CUSTOM = {"\U0001F480": os.path.join(ROOT, "assets", "USER_ADDED", "emoji", "skull.png")}  # owner-supplied skull


def emoji_img(ch, px):
    if ch in CUSTOM and os.path.exists(CUSTOM[ch]):
        im = Image.open(CUSTOM[ch]).convert("RGBA")
        im = im.resize((round(im.width * px / im.height), px), Image.LANCZOS)
        from PIL import ImageFilter
        pad = 4
        out = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0))
        a = Image.new("L", out.size, 0)
        a.paste(im.getchannel("A"), (pad, pad))
        ring = a.point(lambda v: 255 if v > 90 else 0).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(0.6))
        out.paste(Image.new("RGBA", out.size, (40, 40, 48, 255)), (0, 0), ring)
        out.alpha_composite(im, (pad, pad))
        return out
    f = ImageFont.truetype(EMOJI, 109)
    tmp = Image.new("RGBA", (300, 160), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((8, 8), ch, font=f, embedded_color=True)
    tmp = tmp.crop(tmp.getbbox())
    tmp = tmp.resize((round(tmp.width * px / tmp.height), px), Image.LANCZOS)
    # thin dark outline so pale emoji (e.g. the Noto skull) read on the white box
    from PIL import ImageFilter
    pad = 4
    out = Image.new("RGBA", (tmp.width + 2 * pad, tmp.height + 2 * pad), (0, 0, 0, 0))
    a = Image.new("L", out.size, 0)
    a.paste(tmp.getchannel("A"), (pad, pad))
    ring = a.point(lambda v: 255 if v > 90 else 0).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(0.6))
    out.paste(Image.new("RGBA", out.size, (40, 40, 48, 255)), (0, 0), ring)
    out.alpha_composite(tmp, (pad, pad))
    return out


def measure(tokens, f, epx):
    w = 0
    for t in tokens:
        w += sum(emoji_img(c, epx).width + 8 for c in [ch for ch in t if ch not in '\ufe0f\u200d']) if EMO_RE.fullmatch(t) else f.getlength(t)
    return w


def wrap(text, f, epx, max_w):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cur and measure(EMO_RE.split(trial), f, epx) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    return lines


def render(text, out, width=900, size=64):
    f = ImageFont.truetype(FONT, size)
    epx = int(size * 1.18)
    lines = wrap(text, f, epx, width - 60)
    asc, desc = f.getmetrics()
    lh = asc + desc + 22
    canvas = Image.new("RGBA", (1080, lh * len(lines) + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    y = 20
    for ln in lines:
        toks = [t for t in EMO_RE.split(ln) if t]
        w = measure(toks, f, epx)
        x0 = (1080 - w) / 2
        d.rounded_rectangle([x0 - 26, y - 2, x0 + w + 26, y + lh + 2], radius=18, fill=(255, 255, 255, 255))
        y += lh
    y = 20
    for ln in lines:
        toks = [t for t in EMO_RE.split(ln) if t]
        w = measure(toks, f, epx)
        x = (1080 - w) / 2
        for t in toks:
            if EMO_RE.fullmatch(t):
                for c in [ch for ch in t if ch not in '\ufe0f\u200d']:
                    e = emoji_img(c, epx)
                    canvas.alpha_composite(e, (int(x), int(y + (lh - e.height) / 2)))
                    x += e.width + 8
            else:
                d.text((x, y + 10), t, font=f, fill=(0, 0, 0, 255))
                x += f.getlength(t)
        y += lh
    canvas = canvas.crop(canvas.getbbox())
    canvas.save(out)
    print(out, canvas.size, lines)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if "--width" in a:
        kw["width"] = int(a[a.index("--width") + 1])
    if "--size" in a:
        kw["size"] = int(a[a.index("--size") + 1])
    render(a[0], a[1], **kw)
