# def test_generate_report(tmp_path):
#     """Test that generate_report returns valid HTML and saves to a file when requested."""
#     df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
#     html = generate_report(df)
#     assert "<html>" in html

#     file_path = tmp_path / "report.html"
#     generate_report(df, output_file=str(file_path))
#     assert file_path.exists()
