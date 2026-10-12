import urllib.request, os, io, zipfile

YEARS = range(2010, 2027)        # acopera fereastra de test; ajusteaza daca vrei
os.makedirs("odds", exist_ok=True)

def fetch(year):
    base = f"http://www.tennis-data.co.uk/{year}/{year}"
    dest = f"odds/tennis_odds_{year}.xlsx"
    headers = {"User-Agent": "Mozilla/5.0"}
    # 1) incearca .xlsx direct
    try:
        req = urllib.request.Request(base + ".xlsx", headers=headers)
        open(dest, "wb").write(urllib.request.urlopen(req).read())
        return f"OK (xlsx)  {year}"
    except Exception:
        pass
    # 2) fallback: descarca .zip si extrage xlsx-ul
    try:
        req = urllib.request.Request(base + ".zip", headers=headers)
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req).read()))
        name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        open(dest, "wb").write(z.read(name))
        return f"OK (zip)   {year}"
    except Exception as e:
        return f"FAIL       {year}: {e}"

for y in YEARS:
    print(fetch(y))