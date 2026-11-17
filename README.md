# Meme generator

Ubaci svoju sliku na papir koji čovek drži, dopiši natpis i preuzmi meme kao PNG ili JPG.
Sve radi u pregledaču. Nema servera i slike se nigde ne šalju.

Sajt: https://reductioadabsurdum.github.io/meme-generator/

## Lokalno

Otvori `docs/index.html` duplim klikom.

## Objavljivanje (GitHub Pages)

Na GitHubu otvori **Settings → Pages**. Za **Source** izaberi **Deploy from a branch**, za granu **main**, za folder **/docs**, pa klikni **Save**. Posle minut-dva sajt je na adresi iznad.

Ako se adresa promeni (drugo ime repozitorijuma ili svoj domen), u `docs/index.html` promeni `og:url` i `og:image`. Od njih zavisi pregled linka na Facebooku, Viberu i X-u.

## Fajlovi

- `docs/index.html`, `docs/app.js` — sajt
- `docs/assets/template-data.js` — šablon (očišćen kadar, maska, senke i mreža savijanja papira)
- `docs/assets/og.jpg` — slika za pregled linka
- `docs/assets/default-image.js` — primer slika koja stoji na papiru dok posetilac ne izabere svoju
- `tools/prepare_template.py` — od `tools/original-screenshot.png` pravi oba fajla iz `docs/assets/`. Treba ti samo ako menjaš šablon. Potrebni su Python 3, numpy, pillow, scipy i opencv-contrib-python-headless.
- `tools/make_default_image.py` — od `tools/default-photo.jpeg` pravi `docs/assets/default-image.js`. Za drugu primer sliku: `python3 tools/make_default_image.py slika.jpg` (potreban je pillow).
