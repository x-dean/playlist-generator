import click

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--validate', is_flag=True, help='Validate data completeness')
@click.option('--fix', is_flag=True, help='Attempt to fix missing data')
def validate_database(file_path: str, validate: bool, fix: bool):
    """Validate and fix database data completeness."""
    try:
        from ..core.database import get_db_manager
        db_manager = get_db_manager()
        
        # Validate data completeness
        if validate:
            result = db_manager.validate_data_completeness(file_path)
            
            click.echo(f"Database validation for: {file_path}")
            click.echo(f"Overall data quality: {result['data_quality']:.2%}")
            click.echo(f"Valid: {result['valid']}")
            
            if result['missing_fields']:
                click.echo("\nMissing fields by category:")
                for category, fields in result['missing_fields'].items():
                    click.echo(f"  {category}: {', '.join(fields)}")
            
            if result['json_parsing_issues']:
                click.echo(f"\nJSON parsing issues: {', '.join(result['json_parsing_issues'])}")
            
            if result['tag_sources']:
                click.echo(f"\nTag sources: {result['tag_sources']}")
            
            click.echo(f"\nFields present: {result['fields_present']}/{result['total_fields_checked']}")
            
            # Data quality by category
            click.echo("\nData quality by category:")
            for category, quality in result['data_quality_by_category'].items():
                click.echo(f"  {category}: {quality:.2%}")
        
        # Get analysis result with parsed JSON
        analysis_result = db_manager.get_analysis_result(file_path)
        if analysis_result:
            click.echo(f"\nAnalysis result retrieved successfully")
            click.echo(f"Title: {analysis_result.get('title', 'Unknown')}")
            click.echo(f"Artist: {analysis_result.get('artist', 'Unknown')}")
            click.echo(f"Duration: {analysis_result.get('duration', 'Unknown')}")
            
            # Show parsed JSON fields
            json_fields = ['bpm_estimates', 'mfcc_coefficients', 'embedding', 'tags']
            for field in json_fields:
                if field in analysis_result and analysis_result[field]:
                    if isinstance(analysis_result[field], (list, dict)):
                        click.echo(f"{field}: {type(analysis_result[field]).__name__} with {len(analysis_result[field])} items")
                    else:
                        click.echo(f"{field}: {type(analysis_result[field]).__name__}")
        else:
            click.echo(f"No analysis result found for: {file_path}")
            
    except Exception as e:
        click.echo(f"Error: {e}")
        return 1
    
    return 0 