# TalentProof AI — Enterprise Frontend

Bu qovluq **TalentProof AI** layihəsinin korporativ, sübut əsaslı və beynəlxalq standartlara uyğun hazırlanmış frontend tətbiqidir.

---

## Layihə Faylları
- `index.html`: Əsas interfeys (HTML5 + Tailwind CSS CDN).
- `app.js`: İnteraktiv idarəetmə mühərriki, API inteqrasiyası (`localhost:8000`), What-If simulyasiyası və namizəd təhlil matrisi.
- `i18n.js`: Azərbaycan (`AZ`), İngilis (`EN`) və Rus (`RU`) dilləri üçün 100% tam tərcümə lüğəti.
- `style.css`: Qətiyyən emoji və vibe-coded effekti olmayan, korporativ palitraya (`#0F172A`, `#1E293B`, Muted Emerald, Muted Rose) uyğun dizayn sistemi.

---

## Necə İşə Salmaq Olar?
1. **Sadə üsul:** `frontend/index.html` faylını birbaşa istənilən brauzerdə açın (iki dəfə klikləməklə).
2. **Lokal server ilə:**
   ```bash
   npx serve frontend
   ```
   və ya VS Code / Antigravity daxilində **Live Server** uzantısı ilə aça bilərsiniz.

---

## Backend ilə Əlaqə
Frontend avtomatik olaraq `http://localhost:8000` ünvanında işləyən FastAPI backend serveri ilə inteqrasiya edir:
- Backend aktiv olduqda real vaxt rejimində ML modelinin nəticələri gətirilir.
- Backend söndürüldükdə isə frontend fasiləsiz olaraq avtonom lokal rejimdə işləməyə davam edir.
