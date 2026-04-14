import argparse
import os
import re


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Gomill SGF game results to a PGN file for BayesElo."
    )
    parser.add_argument(
        "--games-dir",
        type=str,
        default="my_league_v1.games",
        help="Path to the Gomill games folder containing .sgf files (default: my_league_v1.games).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.pgn",
        help="Output PGN file path for BayesElo (default: results.pgn).",
    )
    return parser.parse_args()


def generate_bayeselo_file(games_dir: str, output_file: str) -> None:
    if not os.path.exists(games_dir):
        print(f"Error: Could not find directory '{games_dir}'")
        return

    sgf_files = [f for f in os.listdir(games_dir) if f.endswith('.sgf')]
    
    if not sgf_files:
        print(f"No .sgf files found in {games_dir}")
        return

    valid_games = 0

    with open(output_file, 'w') as out_file:
        for sgf in sgf_files:
            filepath = os.path.join(games_dir, sgf)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract player names and the result using Regex
            white_match = re.search(r'PW\[([^\]]+)\]', content)
            black_match = re.search(r'PB\[([^\]]+)\]', content)
            result_match = re.search(r'RE\[([^\]]+)\]', content)

            if white_match and black_match and result_match:
                white = white_match.group(1)
                black = black_match.group(1)
                sgf_result = result_match.group(1).upper()

                # CRITICAL: BayesElo is a chess tool. 
                # In chess, 1-0 means White wins, 0-1 means Black wins.
                if sgf_result.startswith('W'):
                    pgn_result = "1-0"  
                elif sgf_result.startswith('B'):
                    pgn_result = "0-1"  
                else:
                    pgn_result = "1/2-1/2" # Draw

                # Write the minimal PGN format required by BayesElo
                out_file.write(f'[White "{white}"]\n')
                out_file.write(f'[Black "{black}"]\n')
                out_file.write(f'[Result "{pgn_result}"]\n\n')
                
                # IMPORTANT FIX: BayesElo needs a dummy move to close the game record
                out_file.write(f'1. e4 {pgn_result}\n\n')
                
                valid_games += 1

    print(f"Success! Converted {valid_games} games into '{output_file}'.")
    print("You can now feed this file directly into BayesElo.")


if __name__ == "__main__":
    args = _parse_args()
    generate_bayeselo_file(args.games_dir, args.output)