from attack import evaluate_e5


def test_no_overlap():
    # Arrange
    references = [["the bear eats in the house"]]
    predictions = ["dog runs away from home"]

    # Act
    score = evaluate_e5.calculate_word_overlap_score(predictions, references)

    # Assert
    assert score == 0.0, "Expected no overlap"


def test_some_overlap():
    # Arrange
    references = [["the bear eats in the house"]]
    predictions = ["the dog eats outside the house"]
    expected_score = 3 / 5  # 3 words overlap, 5 unique words in reference

    # Act
    score = evaluate_e5.calculate_word_overlap_score(predictions, references)

    # Assert
    assert score == expected_score, f"Expected some overlap of {expected_score} words"


def test_full_overlap():
    # Arrange
    references = [["the bear eats in the house"]]
    predictions = ["the bear eats in the house"]

    # Act
    score = evaluate_e5.calculate_word_overlap_score(predictions, references)

    # Assert
    assert score == 1.0, "Expected full overlap"


def test_empty_predictions():
    # Arrange
    references = [["the bear eats in the house"]]
    predictions = []

    # Act
    score = evaluate_e5.calculate_word_overlap_score(predictions, references)

    # Assert
    assert score == 0.0, "Expected zero score when predictions are empty"
