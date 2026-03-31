# AutoTrader — TODO

Dieses TODO startet ab dem aktuellen Projektstand. Der MVP-Loop ist laut `README.md`
bereits vorhanden; offen sind jetzt vor allem Produktionsreife im Paper Trading,
Risikosteuerung und belastbare Vergleichbarkeit neuer Strategien.

## Aktueller Stand

- [x] Exchange Connector + Paper Trading Engine
- [x] Docker-basierter Strategy Runner
- [x] Evolution Loop mit Analyse, Blocks und Wave-Plans
- [x] Hall of Fame, Confidence Scoring und Loop Detection
- [x] Meta-Tracking fuer Fitness-Trend und Modellvergleich

---

## Schritt 0: Baseline — System erst mal laufen lassen

Bevor wir an der Paper-Engine schrauben, brauchen wir eine Baseline.
Ohne die wissen wir nach Track A nicht, ob die Aenderungen wirklich etwas verbessern.

- [ ] End-to-End Soak Test ueber 24-72h mit echter Marktdatenlast starten
- [ ] Baseline-Metriken erfassen: Crashrate, Strategien pro Wave, Fitness-Verteilung, Trade-Frequenz
- [ ] Bekannte Schwaechen und Auffaelligkeiten dokumentieren (z.B. "alle Strategies kaufen sofort")
- [ ] Minimale Unit-Tests fuer `PaperExchange.place_order` schreiben (Buy, Sell, Reject-Pfade)
- [ ] Minimale Unit-Tests fuer `evaluator.evaluate` (Fitness, Confidence, Edge Cases)

---

## Phase 2.5: Jetzt wichtig — Production-Grade Paper Trading

### Track A: Execution Reality Modeling

Das aktuelle `place_order` in `exchange/paper.py` fuellt immer sofort zum Last Price.
In der Realitaet gibt es Spread, Slippage, Mindestgroessen und Rundung.
Ohne das testet jede Strategie gegen eine idealistische Welt und scheitert spaeter live.

> **Referenz:** QuantConnect nennt diesen Bereich "Reality Modeling" und teilt ihn in
> [Slippage Models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts)
> und [Trade Fill Models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts).
> Kernidee: der Fill-Preis ist nie der Last Price; er haengt ab von Spread, Volumen
> und Order-Typ. Selbst ein einfaches konstantes Slippage-Modell (z.B. 0.01% Aufschlag)
> ist besser als kein Modell. Ein Volume-Share-Modell geht weiter, indem es die
> Order-Groesse relativ zum Marktvolumen beruecksichtigt.

#### Welle 1 — Minimum Viable Realism (zuerst)
- [ ] Spread aus Orderbook holen und in `place_order` einpreisen (Buy: Ask, Sell: Bid statt Last)
- [ ] Konfigurierbares Slippage-Modell: `SLIPPAGE_PCT` als konstanter Aufschlag auf den Fill-Preis
- [ ] `MIN_NOTIONAL` erzwingen (Binance BTC/USDT: 5 USDT) — Reject wenn darunter
- [ ] Quantity-Rundung auf gueltige `step_size` / `tick_size` (Binance LOT_SIZE Filter)
- [ ] Rejections mit echten Gruenden: `insufficient_balance`, `min_notional`, `lot_size`
- [ ] Config: `SLIPPAGE_PCT`, `MIN_NOTIONAL`, `TICK_SIZE`, `STEP_SIZE` in `.env` + `config.py`
- [ ] Tests: Slippage wirkt, Spread wirkt, Rejection greift bei zu kleiner Order

#### Welle 2 — Erweiterte Realism (danach, wenn Welle 1 stabil laeuft)
- [ ] Maker/Taker Fee-Modell statt pauschaler Fee (`MAKER_FEE_PCT` / `TAKER_FEE_PCT`)
- [ ] Optionale kuenstliche Latenz zwischen Signal und Fill (`FILL_DELAY_MS`)
- [ ] Optional: Partial Fills und Restmengen (erst wenn echter Bedarf sichtbar wird)
- [ ] Optional: Volume-basiertes Slippage statt konstantem Aufschlag

### Track B: Global Risk Governance
- [ ] Globalen Risk Layer oberhalb einzelner Strategien einfuehren
- [ ] `MAX_NOTIONAL_PER_STRATEGY` und `MAX_PORTFOLIO_EXPOSURE` in Config/.env aufnehmen
- [ ] `MAX_DAILY_LOSS_PCT` und `MAX_DRAWDOWN_PCT` als Kill-Switches einfuehren
- [ ] `MAX_CONSECUTIVE_LOSSES` und `COOLDOWN_MINUTES` fuer automatische Sperren nutzen
- [ ] Harte Guards fuer Positionsgroesse, gleichzeitige Exits/Entries und Notabschaltung bauen
- [ ] Risk-Entscheidungen und Blockierungen sauber loggen

### Track C: Champion / Challenger
- [ ] Hall of Fame zu einem echten Champion/Challenger-Modell ausbauen
- [ ] Shadow Mode: neue Strategien parallel zu Champions laufen lassen, ohne sofort zu "gewinnen"
- [ ] Promotion-Regeln definieren: Mindestdauer, Mindesttrades, Mindestkonfidenz, Risk-Adjusted Score
- [ ] Schlechte ehemalige Champions automatisch degradieren oder retiren
- [ ] Vergleich immer auf gleichen Marktfenstern und gleicher Kapitalbasis erzwingen

### Track D: Observability und Experiment Tracking
- [ ] LLM-Kosten pro Generation/Wave speichern
- [ ] Prompt-Hash, Tokenverbrauch, Antwortlaenge und Latenz pro LLM-Call loggen
- [ ] Container-Metriken sammeln: Crashs, Timeouts, Laufzeit, CPU/RAM grob erfassen
- [ ] Strategy-Run Logs strukturieren (`run_id`, `strategy_id`, `wave_id`, `model_used`)
- [ ] "Warum wurde etwas promoted/rejected?" als nachvollziehbare Audit-Spur speichern

---

## Umsetzungsplan

### 0. Baseline (Voraussetzung fuer alles Weitere)
- System starten, 24-72h laufen lassen, Metriken erfassen.
- Minimale Tests fuer PaperExchange und Evaluator schreiben.
- Erst danach am Execution-Modell aendern.

### 1. Track A Welle 1 — Minimum Viable Realism
- Ziel: Fills sind nicht mehr idealistisch; der Paper-Mode ist eher pessimistisch.
- Betroffene Dateien: `exchange/paper.py`, `config.py`, `.env.example`, `tests/test_paper.py`
- Erfolgskriterium: Kein Trade wird mehr exakt zum Last Price gefuellt.

### 2. Track A Welle 2 — Erweiterte Realism
- Ziel: Fee-Differenzierung, optionale Latenz, Partial Fills bei Bedarf.
- Nur anfangen wenn Welle 1 sauber im Soak Test laeuft.

### 3. Track B — Risk Governance
- Ziel: Keine Strategie kann das Gesamtsystem gefaehrden.
- Betroffene Dateien: neues `risk/governor.py`, `config.py`, `.env.example`, `orchestrator.py`
- Erfolgskriterium: Harte Portfolio-Grenzen greifen auch bei absurden Signalen.

### 4. Track C — Champion / Challenger
- Ziel: Stabile Strategien verteidigen sich gegen neue Kandidaten.
- Betroffene Dateien: `knowledge/store.py`, `evolution/orchestrator.py`
- Erfolgskriterium: Promotion braucht Mindestdauer + Mindesttrades + Risk-Adjusted Score.

### 5. Track D — Observability
- Ziel: Jede Wave ist spaeter technisch und oekonomisch rekonstruierbar.
- Betroffene Dateien: `evolution/generator.py`, `knowledge/store.py`
- Erfolgskriterium: LLM-Kosten und Prompt-Metriken sind pro Wave abrufbar.

### 6. Erst danach die Breite vergroessern.
- Multi-Symbol, neue Datenquellen und spaetere Live-Features nicht auf einer wackligen Basis bauen.

## Phase 3: Danach — Breite und Robustheit ueber mehrere Maerkte

- [ ] WebSocket statt REST-Polling fuer Marktdaten
- [ ] Multi-Symbol Support mit gemeinsamer Kapitalallokation
- [ ] Regime-aware Scoring nach Symbol und Marktphase
- [ ] Crash Recovery / Resume inklusive Wiederherstellung laufender Runs
- [ ] Strategy Lineage Visualization / Dashboard
- [ ] Langlaufende Soak Tests und automatische Health Checks

## Phase 4: Spaeter — Forschung und Alpha-Ausbau

- [ ] Multi-Timeframe Strategien
- [ ] Fortgeschrittene Regime Detection (z.B. HMM, Clustering, probabilistische Regimes)
- [ ] Strategy Ensembles / Abstimmung mehrerer Strategien
- [ ] Funding Rate, Open Interest und Long/Short Ratio gezielt als Feature-Klasse evaluieren
- [ ] On-Chain-Signale nur mit sauberem Ablation-Test einfuehren
- [ ] Orderbook-/Tradeflow-Mikrostruktur als eigene Block-Kategorie ausbauen
- [ ] Asset-Selektion durch KI statt festem `DEFAULT_SYMBOL`

## Phase 5: Noch spaeter — Live Trading

- [ ] Echter Binance-Connector fuer Live Orders
- [ ] Start nur mit Micro-Positionen und konservativen Guards
- [ ] Exchange-native Schutzmechanismen nutzen, wo moeglich
- [ ] Operativer Runbook-Mode: Kill-Switch, Pause, Forced Exit, Recovery
- [ ] Multi-Exchange Support

## Nicht jetzt priorisieren

- [ ] News-/Social-Sentiment erst nach belastbarem Ablation-Framework
- [ ] Vollautonome Web-Recherche pro Wave nicht in den Kernloop einbauen
- [ ] RL / komplexe Agentensysteme erst nach sauberem Execution- und Risk-Fundament

## Entscheidungen

- **Paper Trading**: Eigene Engine mit echten Marktdaten (kein Binance Testnet). Die naechste Ausbaustufe ist nicht mehr "mehr Features", sondern "realistischere Execution".
- **Sandboxing**: Docker-Container statt Subprocess. Strategien haben volle Freiheit, die Host-Seite behaelt Risk Controls.
- **LLM**: LiteLLM fuer Multi-Model. Als naechstes kommt nicht nur "welches Modell gewinnt?", sondern auch "zu welchem Preis?".
- **Loop Detection**: Eigenentwicklung. Die naechste sinnvolle Ergaenzung ist Champion/Challenger, nicht noch mehr Prompt-Komplexitaet.
