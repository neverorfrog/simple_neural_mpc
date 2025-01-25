# TODO

- [ ] Fare una mlp con e una senza skip connection
- [ ] Provare rete neural con e senza skip connection sull'uniciclo cinematico
- [ ] Proviamo la skip connection come residuo e come non residuo


## Con Skip Connection ma senza in fase di inferenza

Se abbiamo la skip connection nella rete, il modello dinamico totale sará
`dyn_model + mlp_output`. Questo vuol dire che in fase di training la skip
connection resta e riceve come punto di partenza proprio `dyn_model`. In
fase di inferenza invece la skip connection non c'é piú, e aggiungiamo a
`dyn_model` aggiungiamo quello che esce dall'mlp.

### Esempio

TODO

## Con Skip Connection e con in fase di inferenza

TODO

### Esempio

TODO

## Senza Skip Connection

Se non abbiamo la skip connection nella rete, di base impariamo tutto il
modello dinamico. Si puó provare sia a imparare il modello cosí, sia a fare
come ha fatto (forse) Salzmann, che impara i residui.

### Esempio

TODO