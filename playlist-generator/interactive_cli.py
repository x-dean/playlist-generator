import os
import sys
import subprocess
import questionary
from questionary import Separator

# Helper to build CLI args
def build_args(options):
    args = [sys.executable, 'main.py']
    if options['mode'] == 'Analyze Only':
        args.append('-a')
    elif options['mode'] == 'Generate Only':
        args.append('-g')
    elif options['mode'] == 'Update':
        args.append('-u')
    elif options['mode'] == 'Status':
        args.append('--status')
    elif options['mode'] == 'Enrich Tags':
        args.append('--enrich_tags')
        if options.get('force_enrich_tags'):
            args.append('--force_enrich_tags')
    # Common options
    if options.get('music_dir'):
        args += ['--music_dir', options['music_dir']]
    if options.get('host_music_dir'):
        args += ['--host_music_dir', options['host_music_dir']]
    if options.get('output_dir'):
        args += ['--output_dir', options['output_dir']]
    if options.get('workers'):
        args += ['--workers', str(options['workers'])]
    if options.get('num_playlists'):
        args += ['--num_playlists', str(options['num_playlists'])]
    if options.get('playlist_method'):
        args += ['-m', options['playlist_method']]
    if options.get('min_tracks_per_genre'):
        args += ['--min_tracks_per_genre', str(options['min_tracks_per_genre'])]
    if options.get('force_sequential'):
        args.append('--force_sequential')
    return args

def main():
    while True:
        # Main menu
        mode = questionary.select(
            "What do you want to do?",
            choices=[
                "Analyze Only",
                "Generate Only",
                "Update",
                "Status",
                "Enrich Tags",
                Separator(),
                "Exit"
            ]).ask()
        if mode == "Exit":
            print("Goodbye!")
            break
        options = {'mode': mode}
        # Common options
        options['music_dir'] = questionary.text(
            "Music directory (container path):",
            default="/music"
        ).ask()
        options['host_music_dir'] = questionary.text(
            "Music directory (host path):",
            default="/music"
        ).ask()
        options['output_dir'] = questionary.text(
            "Output directory:",
            default="/app/playlists"
        ).ask()
        # Workers
        if mode in ("Analyze Only", "Generate Only", "Update"):
            options['workers'] = questionary.text(
                "Number of workers (blank for auto):",
                default=""
            ).ask()
            if options['workers'] == "":
                options['workers'] = None
        # Playlist method
        if mode in ("Generate Only", "Update"):
            options['playlist_method'] = questionary.select(
                "Playlist generation method:",
                choices=["all", "time", "kmeans", "cache", "tags"],
                default="all"
            ).ask()
            if options['playlist_method'] == "kmeans":
                options['num_playlists'] = questionary.text(
                    "Number of playlists (for kmeans):",
                    default="8"
                ).ask()
            if options['playlist_method'] == "tags":
                options['min_tracks_per_genre'] = questionary.text(
                    "Minimum tracks per genre (tags method):",
                    default="10"
                ).ask()
        # Force sequential
        if mode in ("Analyze Only", "Generate Only", "Update"):
            options['force_sequential'] = questionary.confirm(
                "Force sequential processing? (no parallelism)",
                default=False
            ).ask()
        # Enrich tags
        if mode == "Enrich Tags":
            options['force_enrich_tags'] = questionary.confirm(
                "Force re-enrichment of all tags? (overwrite)",
                default=False
            ).ask()
        # Show summary
        print("\nSummary of your choices:")
        for k, v in options.items():
            print(f"  {k}: {v}")
        if not questionary.confirm("Proceed with these options?", default=True).ask():
            print("Cancelled. Returning to main menu.\n")
            continue
        # Build and run command
        args = build_args(options)
        print(f"\nRunning: {' '.join(str(a) for a in args)}\n")
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        print("\nOperation complete.\n")
        if not questionary.confirm("Do you want to run another operation?", default=True).ask():
            print("Goodbye!")
            break

if __name__ == "__main__":
    main() 