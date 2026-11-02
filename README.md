# Keep or Return

Skin AI check-ins inside your return window tell you whether the skincare you just bought is actually working — keep it, or return it before the deadline.

Built for the YouCam API Skin AI & eCommerce VTO Hackathon.

## Run
Open `index.html` in any modern browser (no build step). 
- **Demo mode**: click "Load 3-week demo story" to see the full flow with sample scores.
- **Live mode**: choose "Live — YouCam Skin AI API", paste your YouCam API key, upload a selfie per check-in.

## YouCam API used
YouCam **Skin Analysis API** (s2s v1.1): register file upload → PUT image → create `skin-analysis` task with concerns (acne, redness, pore, wrinkle, moisture, age_spot, oiliness, texture) → poll task → per-concern scores.

## Verdict engine
Tracks only the concern the product claims to fix, fits a trend over check-ins, compares the gain with the user's bar and the days left in the return window, and outputs KEEP / RETURN / KEEP TESTING.

License: MIT
