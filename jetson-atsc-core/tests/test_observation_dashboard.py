from pathlib import Path
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[1]


def test_dashboard_has_distinct_roles_and_no_duplicate_ids():
    html=(ROOT/"src/ui/static/index.html").read_text()
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.roles=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if "id" in attrs:self.ids.append(attrs["id"])
            if "data-camera" in attrs:self.roles.append(attrs["data-camera"])
    parser=Parser();parser.feed(html)
    assert set(parser.roles)=={"system","acquisition"}
    assert len(parser.ids)==len(set(parser.ids))
    for element in ["predictNextGreen","systemFeed","acquisitionFeed","vehicleLedger","cycleComparison","calibrationCanvas"]:
        assert element in parser.ids
    assert "Observation only" in html
    assert "Control delay" in html
