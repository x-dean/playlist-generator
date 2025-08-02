"""
Dependency injection container for Playlist Generator.
Manages service dependencies and lifecycle.
"""

from typing import Type, Dict, Any, Optional
from ..domain.interfaces import ITrackRepository, IAnalysisRepository, IPlaylistRepository, IAudioAnalyzer, IMetadataEnrichmentService


class Container:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, Type] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, callable] = {}
    
    def register_singleton(self, interface: Type, implementation: Type):
        """Register a singleton service."""
        self._services[interface] = implementation
    
    def register_factory(self, interface: Type, factory: callable):
        """Register a factory function for service creation."""
        self._factories[interface] = factory
    
    def register_instance(self, interface: Type, instance: Any):
        """Register an existing instance."""
        self._singletons[interface] = instance
    
    def resolve(self, interface: Type) -> Any:
        """Resolve a service instance."""
        # Return existing singleton
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Use factory if available
        if interface in self._factories:
            instance = self._factories[interface](self)
            self._singletons[interface] = instance
            return instance
        
        # Create new instance
        if interface in self._services:
            implementation = self._services[interface]
            instance = self._create_instance(implementation)
            self._singletons[interface] = instance
            return instance
        
        raise ValueError(f"No registration found for {interface}")
    
    def _create_instance(self, implementation: Type) -> Any:
        """Create instance with dependency injection."""
        import inspect
        
        # Get constructor signature
        sig = inspect.signature(implementation.__init__)
        params = {}
        
        # Resolve dependencies
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Skip optional parameters
            if param.default != inspect.Parameter.empty:
                continue
            
            # Resolve parameter type
            if param.annotation != inspect.Parameter.empty:
                try:
                    params[param_name] = self.resolve(param.annotation)
                except ValueError:
                    # Skip if dependency not registered
                    pass
        
        return implementation(**params)
    
    def clear(self):
        """Clear all registrations."""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = configure_container()
    return _container


def configure_container() -> Container:
    """Configure the dependency injection container."""
    container = Container()
    
    # Register repositories
    from .repositories import SQLiteTrackRepository, SQLiteAnalysisRepository, SQLitePlaylistRepository
    container.register_singleton(ITrackRepository, SQLiteTrackRepository)
    container.register_singleton(IAnalysisRepository, SQLiteAnalysisRepository)
    container.register_singleton(IPlaylistRepository, SQLitePlaylistRepository)
    
    # Register services
    from .services import EssentiaAudioAnalyzer, MusicBrainzEnrichmentService
    container.register_singleton(IAudioAnalyzer, EssentiaAudioAnalyzer)
    container.register_singleton(IMetadataEnrichmentService, MusicBrainzEnrichmentService)
    
    # Register configuration
    from .config import ConfigurationService
    config_service = ConfigurationService()
    container.register_instance(ConfigurationService, config_service)
    
    return container 