from django import forms


class TrainingForm(forms.Form):
    epochs = forms.IntegerField(min_value=1, max_value=20, initial=1)
    batch_size = forms.IntegerField(min_value=1, max_value=8, initial=1)
    gradient_accumulation = forms.IntegerField(min_value=1, max_value=64, initial=8)
    max_length = forms.TypedChoiceField(choices=[(128, "128"), (256, "256"), (512, "512")], coerce=int, initial=256)
    learning_rate = forms.FloatField(
        min_value=0.000001, max_value=0.001, initial=0.00008,
        help_text="Kept constant for the entire training run.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-select" if isinstance(field.widget, forms.Select) else "form-control"


class DialogueForm(forms.Form):
    run_id = forms.ChoiceField(label="Trained model")
    persona = forms.CharField(label="Assistant persona", max_length=2000,
                              initial="I like to remodel homes.\nI like to go hunting.",
                              help_text="Describe the assistant's character, one fact per line.",
                              widget=forms.Textarea(attrs={"rows": 3}))
    message = forms.CharField(max_length=2000, widget=forms.Textarea(attrs={"rows": 2}))

    def __init__(self, *args, available=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["run_id"].choices = [(run["id"], run["id"]) for run in available]
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-select" if isinstance(field.widget, forms.Select) else "form-control"
