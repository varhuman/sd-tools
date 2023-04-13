import gradio as gr

def closest_match(x):
    return x + ": The Definitive Edition"

def Dropdown_list(x):
    new_options =  [*options, x + " Remastered", x + ": The Remake", x + ": Game of the Year Edition", x + " Steelbook Edition"]
    return gr.Dropdown.update(choices=new_options)


def Recommend_new(x):
  return x + ": Highest Cosine Similarity"

demo = gr.Blocks()

options = ['Placeholder A', 'Placeholder B', 'Placeholder C']
with demo:

    text_options = gr.Dropdown(options, label="Top 5 options")
    
    new_title = gr.Textbox(label="Here you go!")
    b3 = gr.Button("Recommend a new title")

    b3.click(Recommend_new, inputs=text_options, outputs=new_title)
    # text_options.update(interactive=True)


demo.launch(debug=True)