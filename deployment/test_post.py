import requests


def test_post():
    data = {
        "competition_id": 24,
        "home_club_id": 60,
        "away_club_id": 40,
        "home_club_position": 12,
        "away_club_position": 8,
    }

    print(data)

    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    print(response.json())
