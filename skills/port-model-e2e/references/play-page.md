# Phase 8 — a play page for the port

The Reactor Sandbox drives any model from its schema, so a person can always reach the port. A play
page goes further: it is this model's own controls, with the baseline's examples loaded in one
click, so the person tests the model instead of filling in fields. It runs on the node, next to the
served port, and the report tells the person how to open it from where they sit.

The page is a local tool. It talks only to the model the person is running, at an address they
give it. It has no hosted endpoint, no API key, no token route and no knowledge of any deployment.

## 1. The controls come from the served schema

Read the schema the served port renders (`GET /schema` on the serve port, or the file Phase 4
wrote). Its `paths` are the commands, its `webhooks` the messages the model sends, its
`x-reactor.tracks` the media. Build the page from those names and value sets only. A control the
schema does not list does not exist, however plainly the source's README describes it.

| What the schema says | What to render |
|---|---|
| a small set of choices | segmented buttons, or a key pad when the choices are directions |
| a number with bounds | a slider labelled with its real unit, and a number box |
| free text | a text field that grows to fit, with an explicit Send |
| a file | an upload with a preview |
| a boolean | a toggle |
| a point or a drag in the frame | a click or drag on the display, converted to the frame's pixels and origin as the command's description states |
| no arguments | a button |

Every key binding has an on-screen control that does the same thing.

## 2. The page has one layout

- **A connection bar**: status, Connect / Disconnect, and the model URL in one row.
- **The display**: the output tracks from `x-reactor.tracks`, with the empty, loading and stopped
  states shown inside it.
- **A sidebar, in this order**: examples, then the inputs **set before play**, then the prompt,
  then a collapsed Advanced section (seed and rarely used settings). The inputs set before play
  are locked while a session runs and unlock on reset.
- **A controls card under the display**: a telemetry row (chunk, frames, delivered fps, the
  current action), the inputs **changed during play**, and Reset.
- **Refusals** appear where the person is looking, each code in plain words. The client SDK's
  `sendCommand` does not reject on a refused command; it emits an `error` event carrying the
  command and the code, so listen there.

For every input changed during play, show both what was sent and what the model reports using.
A command applies at the next step, so the two differ for a step or two; that lag tells the person
the command arrived, and without it a slow model looks like a dropped command. Reset sends the
model's reset command, clears the display at once and keeps the connection.

The page works at phone width without horizontal scrolling.

## 3. The examples are the baseline's

Ship the baseline's conditions as the page's examples: its input images copied into the page's
files, its prompt and its action or force values. One click loads an example through the same code
path an uploaded file takes, so the shortcut cannot drift from the real path. Add an upload for the
person's own inputs. Never add an example that the baseline did not run.

## 4. The stack and the address

- A standalone web app whose root route is this model, using `@reactor-team/js-sdk` at the latest
  version on npm (`npm view @reactor-team/js-sdk version`). An SDK older than the runtime fails
  with 404s on session routes and "failed to send SDP offer".
- The default model URL is **the page's own hostname on the serve port**
  (`${location.protocol}//${location.hostname}:<serve port>`), overridable by a `?local=` query
  parameter and by the URL field. The same page then works opened over a VPN at
  `http://<node>:<page port>` and through an SSH tunnel at `http://localhost:<page port>`, with
  nothing to edit. No node address is written into the code.
- It runs on the node in a container you built from a Dockerfile in the page's directory, like
  every other process of this run, with `--network host`, listening on all interfaces. Run the
  production build (`build`, then `start`); a build run while a dev server is live overwrites the
  dev server's output and leaves an unstyled page.

## 5. Verify it on the node before handing it over

Drive the page headless on the node itself (Playwright with Chromium, in a container), where page,
signalling and media all reach `localhost`:

1. open the page, connect, load each example, start;
2. count decoded frames (`requestVideoFrameCallback`, or `getVideoPlaybackQuality().totalVideoFrames`)
   over wall-clock time; a live track's `currentTime` advances even when no frame arrives, so it
   proves nothing;
3. change each input changed during play while it runs and confirm the next frames keep arriving;
4. send one command that must be refused and confirm the refusal shows;
5. reset, and confirm the display clears and the connection stays.

Whether the motion follows the inputs is the person's call; the browser test shows that the page
drives the model, not that the model behaves.

## 6. Tell the person how to open it

A browser reaches three things on the node: the page (HTTP on the page port), the model's
signalling (HTTP on the serve port) and the video (WebRTC over UDP, [serving.md](serving.md) §4).
The report gives the case that applies:

| Where the person is | What they do | The video |
|---|---|---|
| On the node's network or VPN | open `http://<node>:<page port>` | arrives directly |
| SSH reaches the node, and its UDP range is open to them | `ssh -N -L <page port>:localhost:<page port> -L <serve port>:localhost:<serve port> <node>`, then open `http://localhost:<page port>` | arrives directly over UDP |
| Only SSH reaches the node | the same tunnel | cannot cross the tunnel: it needs a TURN server both ends reach (`TURN_SERVERS`, `ICE_TRANSPORT_POLICY=relay`), or a machine on the node's network |

A page that connects but stays black has working signalling and blocked media: name which case
applies. The Reactor Sandbox works through the same paths and remains a fallback.

## Exit

- the page is served on the node, and a headless browser on the node connected with no manual
  configuration, loaded every example, started, and counted decoded frames;
- every command the schema lists is reachable on screen;
- the shipped files hold no node address, no hosted endpoint and no API key
  (`grep -rniE "api\.reactor|api_key|token|<node address>"` over them is empty);
- the report states the page's address and the access case, with its commands.
