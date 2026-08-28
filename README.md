# Portfolio

Personal portfolio site — static HTML/CSS/JS, no build step.

## Run locally

Open `index.html` directly, or serve it so relative paths behave the same as production:

```bash
python -m http.server 5500
```

Then visit `http://localhost:5500`.

## Deploy to GitHub Pages

1. Create a new GitHub repo (e.g. `portfolio`) and push this folder to it:

   ```bash
   git init
   git add .
   git commit -m "Initial portfolio"
   git branch -M main
   git remote add origin https://github.com/ShangChiOne/portfolio.git
   git push -u origin main
   ```

2. In the repo on GitHub: **Settings → Pages → Source → Deploy from a branch**, pick `main` and `/ (root)`.
3. The site goes live at `https://shangchione.github.io/portfolio/` a minute or two later.

To use a custom domain, add a `CNAME` file with the domain name and point its DNS at GitHub Pages.

## Structure

- `index.html` — page content
- `styles.css` — all styling
- `script.js` — mobile nav, video modal, scroll reveal
- `assets/images/` — profile photo and project thumbnails
  - `assets/images/fyp/` — real chart exports captured from an actual run of the
    ForecastIQ app (the full walkthrough is a YouTube embed instead, see index.html)
- `FYP Project - TP070073/` — source for the featured ForecastIQ project (see below)

## About the FYP folder

`FYP Project - TP070073/` is the actual Flask app source, not part of the deployed
static site — the portfolio only shows a description card for it. Its `models/` and
`dataset/` folders are large (`rf_model.pkl` alone is ~187MB), and GitHub rejects any
file over 100MB, so `.gitignore` excludes them from this repo.

If you want the FYP source itself on GitHub, push it as its **own** repo (ideally with
[Git LFS](https://git-lfs.com/) for the model files) rather than inside the portfolio
repo, then link to it from the project card.
