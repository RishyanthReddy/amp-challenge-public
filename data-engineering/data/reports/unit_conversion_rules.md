# MIC unit conversion rules

Raw MIC text and units are always retained. Numeric parsing records `=`, `<`, `>`, or `range`; no exact value is invented for censored/range measurements. `µM` is retained unchanged. `µg/mL` and `mg/L` are converted using `µM = (mass concentration in µg/mL × 1000) / molecular_weight_Da`; `mg/mL` uses ×1,000,000. Molecular weight is the sum of standard residue masses plus 18.0153 Da for free termini. Conversion is emitted only with a recognized unit, parsable value, and valid canonical sequence.
