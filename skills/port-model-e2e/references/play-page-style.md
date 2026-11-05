# The play page's look

Every play page uses the same shell and the same visual language, so a person who has played one
ported model knows where everything is on the next. The page is self-contained: these rules need no
component library beyond the stack in [play-page.md](play-page.md) §4. Tailwind classes are given
where the exact value matters.

## Shell

1. **Page.** A full-viewport black column (`h-screen flex flex-col overflow-hidden`). Behind it, two
   `pointer-events-none` layers: a 28px dot grid (1px dots at `rgba(255,255,255,0.07)`) and a soft gold
   radial glow at the top (`rgba(199,192,153,0.12)`, ellipse 80%×50% at 50% −10%). Content layers are
   `relative z-10`. Dark theme only.
2. **Header.** Inside `px-4 sm:px-6 py-3`: a rounded-12 `#1a1a1a` pill with a `white/10` border
   holding the model's name in mono uppercase (`text-xs tracking-widest text-white/50`). Nothing else.
3. **Connection bar, the one control row.** Full-bleed, `px-4 py-2`, background `white/4%`, bottom
   border `white/6%`. Left: an 8px status dot (green connected, yellow pulsing while connecting,
   `white/30` otherwise), a mono `text-xs white/50` status label, one `h-7 px-3 font-mono text-xs`
   Connect / Cancel / Disconnect button, then any connection error in small red text. Right: the model
   URL as a small label and an `h-7` field. The bar wraps on a phone; it never becomes a second bar.
4. **Main area.** `flex-1 min-h-0`, padding `px-4 sm:px-6 pt-3 pb-4 sm:pb-6`, content centred in
   `max-w-7xl`. Desktop (`lg`): two columns, `gap-6`, each scrolling on its own; the page never
   scrolls. Phone: one scrolling stack, `gap-4`, in the order display, controls, sidebar.

## Sidebar: text and session setup

5. **One card**, `lg:w-[min(100%,380px)]`, `shrink-0`, `rounded-xl border border-white/[0.08]
   bg-white/[0.02] p-3 sm:p-4`. Sections are separated by `border-t border-white/[0.06]`; each opens
   with a heading (`font-mono text-xs uppercase tracking-widest`, gold) and one helper line
   (`text-[10px] leading-snug text-white/40`).
6. **Order:** Examples → session setup inputs → Prompt → ▸ Advanced. A prompt is a sidebar section,
   never a row in the controls card. Nothing sits below a text area that grows, except a collapsed
   section; a text box's own actions (Send, Clear) go in its label row above it. Errors sit above the
   prompt.
7. **Example cards** stack with `gap-2`: `rounded-lg border border-white/10 bg-white/[0.03]`, hover
   `/[0.06]` and `/20`; a 96×56 thumbnail (`w-24 h-14 rounded-md`); title mono `text-sm` white;
   description mono `text-[10px] white/50`, clamped. The active card gets an amber border, tint and dot;
   a loading card a spinner on its thumbnail. "+ Add your own" comes after the list, never on top.
8. **Text areas show their whole content.** Full width, about 3 rows to start, growing with their text
   (re-measured on change and resize) up to about 45vh, then scrolling. No resize handle.
9. **▸ ADVANCED** holds rarely touched inputs (seed and its Random, sampling settings), collapsed.
   Toggle mono `text-[10px] uppercase tracking-wider white/40`; body `rounded border-white/10
   bg-white/[0.02] p-3 gap-3`; each row a `w-20` label, an `h-7` control, then its value.

## Display and controls

10. **Display.** `relative bg-black rounded-xl border border-white/[0.08] overflow-hidden aspect-video
    w-full`, video `object-contain`. Empty, loading, paused and stopped states sit inside it, centred,
    in small mono text. The display takes no clicks unless the model has a pointer input; input comes
    from the keyboard and the on-screen controls.
11. **Controls card** under the display, same card style as the sidebar. Top to bottom: a telemetry
    row (mono `text-[10px] white/40`: chunk counter, last step time, delivered fps, the active action
    in amber; a spacer; then a Pause / Resume toggle (`set_paused`) and Reset side by side, Reset red-tinted and disabled while
    nothing is running); the live controls, in clusters of `grid-cols-2`, each a centred column with a
    `text-[9px] uppercase tracking-wider white/40` caption over its control.
12. **Keys.** Flat 40×40 `rounded border bg-white/5 border-white/15 text-white/80`, mono `text-xs`.
    Pressed: `bg-amber-300/20 border-amber-300/60 text-amber-200 scale-95`. Disabled: `opacity-30`,
    and a key held when it becomes disabled is released. Movement pads use the inverted-T layout with
    2px gaps; extra keys go below (Space as a wide bar), so each cluster fits half a phone's width.

## Type, colour, density

13. **Type.** Everything interactive or informational is mono: titles 14px, controls 12px, labels and
    telemetry 10px, captions 9px. Uppercase with wide tracking only for headings, captions and buttons.
14. **Colour.** White at graded opacities carries the hierarchy, on black. Gold (`#c7c099`) for
    section headings and the primary button; amber for active, pressed and held; green for connected
    or present; red tints for destructive actions and errors (`bg-red-500/10 border-red-500/30
    text-red-300`).
15. **Density.** 12–16px card padding, 16px between blocks, 8–12px inside a row, `h-7` controls in
    bars and dense rows. Errors inline as small bordered red boxes, never modals. One breakpoint (`lg`);
    rows wrap rather than overflow; no horizontal scroll at 390px.

## Behaviour that shapes the layout

16. **Held inputs are held.** A key or chip applies while held and returns to its base on release:
    pointer up or cancel, leaving the control with the button down, key up, window blur, switching
    examples, disconnect.
17. **Every input change is visible the moment it happens.** The prompt section opens with the text
    in effect (read-only) and its status from sent to reported in use; the editable prompt sits below
    it, and the reported value never overwrites a draft. Held keys show in the telemetry row.
18. **Reset clears the model and the screen, and keeps the connection.** It sends the model's reset
    command and, at once, blanks the display under the empty state, clears held inputs and returns
    the page to "pick an example". Frames already in flight are ignored until a new run's first frame.
