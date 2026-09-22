import marimo

__generated_with = "0.24.2"
app = marimo.App(auto_download=["html"])

with app.setup:
    # Initialization code that runs import marimo as mobefore all other cells
    import marimo as mo


@app.cell
def _():
    mo.md("""
    Welcome !
    """)
    return


if __name__ == "__main__":
    app.run()
