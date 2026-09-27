"""Environment-independent decision overlays and a self-contained replay viewer.

Adapters supply RGB game frames, display metrics, and decision events containing
state, question, options (label/effect/probability), chosen goal, latency and exact
scoring prompts. No model or game imports belong in this module.
"""
import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BG = '#0b1020'
PANEL = '#141d30'
TEXT = '#ecf2fc'
MUTED = '#a9bad3'
ACCENT = '#65e2b2'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(size):
    return ImageFont.truetype(FONT, size)


def wrapped(draw, text, xy, width, size=18, fill=TEXT, spacing=5):
    x, y = xy
    f = font(size)
    for para in text.splitlines():
        line = ''
        for word in para.split():
            candidate = (line + ' ' + word).strip()
            if line and draw.textlength(candidate, font=f) > width:
                draw.text((x, y), line, font=f, fill=fill)
                y += size + spacing
                line = word
            else:
                line = candidate
        draw.text((x, y), line, font=f, fill=fill)
        y += size + spacing
    return y


def png_url(im):
    b = io.BytesIO()
    im.save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def write_viewer(path, meta, events, frames):
    payload = json.dumps({'meta':meta,'events':events,'frames':frames}).replace('</','<\\/')
    template = Path(__file__).with_name('decision_viewer.html').read_text()
    Path(path).write_text(template.replace('__REPLAY_DATA__', payload))
