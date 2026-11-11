# dev notes

Running log of what I saw on real footage and what I changed. Newest at the bottom.

## 2026-11-01

First real clip: Adesanya vs Pereira 2, 8:40-10:10 of the free fight upload. yolo26n-pose on MPS, ~2 min for 90s.

- names came out swapped. red is whoever is on the left at frame 0, that was Izzy here.
- way too sensitive. hand movements / feints logged as missed body punches. added an extension gate
  (wrist has to actually be far from the shoulder at impact) and a retraction check.
- leg kicks firing when they just walk toward each other. ankle moves toward the opponent at step speed,
  shin is right there, every check passes. fix: kicking foot has to be above the other ankle. a step isn't.
- around 1:00 they collide and the skeletons swap permanently. position matching is a coin flip when the
  boxes overlap. added colour histograms of shorts + torso, 60/40 appearance/position, plus an offline
  viterbi pass over the whole clip. position-only on the synthetic collision test: 380/380 frames wrong after.
  with appearance: 0.
- 7th and 8th "Pereira" punches at ~1:05 were Izzy's. pose model attached Izzy's glove to Pereira's skeleton.
  shows up as a one-frame teleport of the wrist keypoint. require the speed peak to have neighbours.
- 1:18 flurry: Pereira 14 -> 23 in 4s for maybe 4 real strikes. same hand twice in 4 frames, one leg kick
  counted twice 6 frames apart, head "reaction" of 1.68 torso/frame (keypoint jump). real strikes in that
  window were 0.43-0.68 speed, the junk was 0.14-0.20. raised the floor, split cooldowns per hand/leg.
- then it was too stingy. teep at 0:01 not counted at all, body punches missing. candidates.csv showed the
  teep rejected with ext 0.87 - impact window caught the leg still chambered. impact is now closest approach
  to the opponent, not max extension. body shots go guard -> ribs with almost no change in extension so the
  retraction check killed them. switched punches to a travel check like kicks.
- missed teep counted as landed. 2D overlap. body kicks now need the opponent's hips to actually move.
- phantom body shot on a camera cut. EMA smoothing turned the cut into a 4-frame "swing". detect cuts from
  the image diff, reset smoothing there, ignore 10 frames either side.

## 2026-11-10

Cut a 19s clip of the round 2 finish for a post.

- after Pereira goes down the blue skeleton jumps onto the ref. he's the second biggest person in frame
  once Pereira is horizontal. if a detection looks like neither fighter and isn't where he was, drop it.
- the KO punch was labelled BODY SHOT. Pereira was bent over so his shoulders were at head height and the
  zone rule put the glove in the body zone. if the glove touches the head sphere it's a head shot, full stop.
- stat panel sits on top of the Paramount+ watermark. bumped opacity, added a top-right option.

still to do before I trust the numbers: run a full round and compare to ufcstats.com.

## 2026-11-10, later

- re-cut of the old render still had the ref problem, and looking at it again I didn't trust the colour
  histogram to separate Pereira's black shorts from the ref's black pants. added a blunt rule on top:
  fighters are shirtless. fraction of the torso crop that isn't skin-hued. first try used brightness and
  flagged Izzy at 0.90 in a backlit frame (median V=31, dark skin in shadow is just dark). hue works:
  brown is dark red, black fabric under arena lights is blue-grey. ref 0.91-0.99, Izzy 0.10-0.15,
  Pereira 0.32 because of the chest tattoos. threshold 0.7.
