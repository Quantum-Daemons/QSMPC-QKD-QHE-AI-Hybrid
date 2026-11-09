import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="5G-NIDD: the flows it holds twice")


@app.cell
def _():
    import json
    import sys

    import marimo as mo

    return json, mo, sys


@app.cell
def _(mo):
    mo.md(r"""
    # Check it yourself: the flows 5G-NIDD holds twice

    This page runs Python in your browser, on [Pyodide](https://pyodide.org), served from this site. Nothing you do here
    leaves the tab.

    The telecom page of this demo says that the published 5G-NIDD file holds the same 281,529 flows twice, labelled Benign in
    one base station's part of the file and UDPFlood in the other's. That claim sets the ceiling on every score reported on
    those rows, so here it is to check. The file is 472 MB, so this page reads 200 evenly spaced pairs of those flows, on all
    113 of the file's columns, as the file's own strings.
    """)
    return


@app.cell
def _(json, mo, sys):
    location = mo.notebook_location() / "public" / "nidd_copies_excerpt.json"
    if sys.platform == "emscripten":
        from pyodide.http import open_url

        excerpt = json.loads(open_url(str(location)).read())
    else:
        excerpt = json.loads(location.read_text())
    columns = excerpt["columns"]
    benign, udpflood = excerpt["labelled_benign"], excerpt["labelled_udpflood"]
    return benign, columns, excerpt, udpflood


@app.cell
def _(benign, mo):
    pair = mo.ui.slider(start=0, stop=len(benign) - 1, value=0, step=1, label="Which of the 200 pairs to read", show_value=True)
    pair
    return (pair,)


@app.cell
def _(benign, columns, mo, pair, udpflood):
    a, b = benign[pair.value], udpflood[pair.value]
    shown = ("Attack Type", "Label", "Attack Tool", "SrcAddr", "DstAddr", "StartTime", "LastTime", "Dur", "Proto", "Sport", "Dport", "TotPkts", "TotBytes",
             "Unnamed: 0", "Rank", "Seq", "Offset", "IdleTime")
    mo.ui.table([{"column": c, "first base station's file": a[columns.index(c)], "second base station's file": b[columns.index(c)]} for c in shown],
                selection=None, pagination=False, label=f"Pair {pair.value}: the columns that decide the question")
    return


@app.cell
def _(benign, columns, mo, udpflood):
    differs = {}
    for k, c in enumerate(columns):
        count = sum(x[k] != y[k] for x, y in zip(benign, udpflood, strict=True))
        if count:
            differs[c] = count
    mo.md(f"""
    ## Column by column, over all {len(benign)} pairs

    Of the {len(columns)} columns, **{len(columns) - len(differs)} are identical in every pair**. The ones that are not:

    {mo.as_html(mo.ui.table([{"column": c, "pairs that differ": n} for c, n in differs.items()], selection=None, pagination=False))}

    Those are the three labels and the columns that say where a row sits in its file. Everything that describes the flow agrees.
    """)
    return (differs,)


@app.cell
def _(excerpt, mo):
    ceiling = 1 - excerpt["pairs_in_file"] / excerpt["rows_in_file"]
    mo.md(rf"""
    ## What that costs

    Rows with the same feature vector get the same prediction, so at most one copy of each pair can be scored right:

    $$\text{{accuracy}} \le 1 - \frac{{{excerpt["pairs_in_file"]:,}}}{{{excerpt["rows_in_file"]:,}}} = {ceiling:.5f}$$

    The classifier on the telecom page scores 0.7616 on the rows as published and 0.9979 with the copies labelled Benign removed. A
    score above {ceiling:.4f} on the rows as published can only come from a column that differs between the copies, and those
    are the ones listed above.
    """)
    return


@app.cell
def _(excerpt, mo):
    mo.md(f"""
    ## Where this came from

    {excerpt["attribution"]} Licence: [{excerpt["licence"]}]({excerpt["licence_url"]}). Changes: {excerpt["modifications"]}

    The other half of the evidence is not in this excerpt: the 421,471 packets behind these flows carry identical timestamps in
    the two capture files, and in each attack file exactly one address pair carries the attack label. The account is in the
    project's `references/5g-nidd-metric-profile.md`. The software this page runs on is listed in
    [THIRD_PARTY.txt](./THIRD_PARTY.txt).
    """)
    return


if __name__ == "__main__":
    app.run()
