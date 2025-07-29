# Migration Plan: Port Missing Features to Refactored Code

## Overview
This plan outlines the migration of missing features from the pre-refactored codebase (`playlist_generator/`) to the refactored codebase (`playlist_generator_refactored/`).

## Phase 1: Tag-Based Playlist Generation

### 1.1 Analysis of Original Implementation
**File**: `playlist_generator/app/playlist_generator/tag_based.py`
**Key Features**:
- MusicBrainz integration for genre enrichment
- Last.fm integration for additional metadata
- Genre-based grouping with minimum track thresholds
- Decade-based grouping
- Mood-based categorization
- Complex playlist naming with genre/decade/mood combinations

### 1.2 Migration Steps
1. **Create Tag-Based Service** (`src/application/services/tag_based_service.py`)
   - Port `TagBasedPlaylistGenerator` class
   - Adapt to new domain entities (`AudioFile`, `Playlist`)
   - Use existing external API clients (`musicbrainz_client.py`, `lastfm_client.py`)

2. **Update Playlist Generation Service**
   - Add `_generate_tag_based_playlist()` method
   - Handle genre, decade, and mood-based grouping
   - Implement minimum track thresholds

3. **Add Configuration**
   - Add tag-based settings to `AppConfig`
   - Configure minimum tracks per genre
   - Set up API keys for external services

### 1.3 Implementation Priority: HIGH
**Estimated Effort**: 2-3 days
**Dependencies**: External API clients already exist

---

## Phase 2: Cache-Based Playlist Generation

### 2.1 Analysis of Original Implementation
**File**: `playlist_generator/app/playlist_generator/cache.py`
**Key Features**:
- Rule-based feature categorization (BPM, energy, mood)
- Cache-based playlist generation from database
- Feature binning with descriptions
- Playlist merging and optimization
- Performance optimization through caching

### 2.2 Migration Steps
1. **Create Cache-Based Service** (`src/application/services/cache_based_service.py`)
   - Port `CacheBasedGenerator` class
   - Adapt feature categorization logic
   - Implement playlist merging algorithms

2. **Update Playlist Generation Service**
   - Add `_generate_cache_based_playlist()` method
   - Implement feature binning logic
   - Add playlist merging functionality

3. **Add Configuration**
   - Define BPM ranges and energy levels
   - Configure mood categories
   - Set up cache optimization settings

### 2.3 Implementation Priority: HIGH
**Estimated Effort**: 2-3 days
**Dependencies**: Database repository pattern

---

## Phase 3: Advanced Playlist Models

### 3.1 Analysis of Original Implementation
**File**: `playlist_generator/app/playlist_generator/advanced_playlist_models.py`
**Key Features**:
- Ensemble clustering (K-means + DBSCAN + Hierarchical)
- Recommendation-based playlists
- Mood-based playlist generation
- Hybrid approaches combining multiple methods
- Sophisticated feature weighting

### 3.2 Migration Steps
1. **Create Advanced Models Service** (`src/application/services/advanced_models_service.py`)
   - Port `AdvancedPlaylistModels` class
   - Implement ensemble clustering methods
   - Add recommendation algorithms

2. **Update Playlist Generation Service**
   - Add `_generate_advanced_playlist()` method
   - Implement ensemble voting system
   - Add mood-based generation

3. **Add Configuration**
   - Configure feature weights
   - Set up clustering parameters
   - Define ensemble methods

### 3.3 Implementation Priority: MEDIUM
**Estimated Effort**: 3-4 days
**Dependencies**: scikit-learn, numpy, pandas

---

## Phase 4: Feature Group Playlist Generation

### 4.1 Analysis of Original Implementation
**File**: `playlist_generator/app/playlist_generator/feature_group.py`
**Key Features**:
- Feature-based grouping using multiple audio characteristics
- BPM-based categorization
- Spectral feature analysis
- Energy level grouping

### 4.2 Migration Steps
1. **Create Feature Group Service** (`src/application/services/feature_group_service.py`)
   - Port feature grouping logic
   - Implement BPM and spectral categorization
   - Add energy level analysis

2. **Update Playlist Generation Service**
   - Add `_generate_feature_group_playlist()` method
   - Implement feature-based clustering
   - Add spectral analysis

### 4.3 Implementation Priority: MEDIUM
**Estimated Effort**: 1-2 days
**Dependencies**: Audio analysis service

---

## Phase 5: Mixed Playlist Generation

### 5.1 Analysis of Original Implementation
**File**: `playlist_generator/app/playlist_generator/playlist_manager.py`
**Key Features**:
- Combination of multiple generation methods
- Hybrid playlist creation
- Method selection based on available data

### 5.2 Migration Steps
1. **Update Playlist Generation Service**
   - Add `_generate_mixed_playlist()` method
   - Implement method combination logic
   - Add intelligent method selection

2. **Add Configuration**
   - Define method combination rules
   - Configure fallback strategies

### 5.3 Implementation Priority: LOW
**Estimated Effort**: 1 day
**Dependencies**: All other playlist methods

---

## Phase 6: Advanced CLI Features

### 6.1 Workers Configuration
**Migration Steps**:
1. Update CLI interface to support `--workers` parameter
2. Add parallel processing configuration
3. Implement worker pool management

### 6.2 Memory Management
**Migration Steps**:
1. Add memory monitoring to services
2. Implement resource optimization
3. Add memory-aware processing modes

### 6.3 Database Diagnostics
**Migration Steps**:
1. Create diagnostic service (`src/application/services/diagnostic_service.py`)
2. Add database health checks
3. Implement performance monitoring

---

## Implementation Order

### Week 1: Core Features
1. **Day 1-2**: Tag-based playlist generation
2. **Day 3-4**: Cache-based playlist generation
3. **Day 5**: Testing and integration

### Week 2: Advanced Features
1. **Day 1-3**: Advanced playlist models
2. **Day 4**: Feature group generation
3. **Day 5**: Mixed generation and testing

### Week 3: Polish and Optimization
1. **Day 1-2**: CLI enhancements
2. **Day 3**: Memory management
3. **Day 4-5**: Database diagnostics and testing

---

## Testing Strategy

### Unit Tests
- Test each playlist generation method independently
- Mock external API calls
- Verify feature extraction and categorization

### Integration Tests
- Test full pipeline with each method
- Verify database persistence
- Test CLI commands with new methods

### Performance Tests
- Compare performance with original implementation
- Test memory usage under load
- Verify caching effectiveness

---

## Success Criteria

### Phase 1-2 (High Priority)
- ✅ Tag-based playlist generation working
- ✅ Cache-based playlist generation working
- ✅ All CLI commands functional
- ✅ Database integration complete

### Phase 3-4 (Medium Priority)
- ✅ Advanced models implemented
- ✅ Feature group generation working
- ✅ Ensemble methods functional
- ✅ Performance optimized

### Phase 5-6 (Low Priority)
- ✅ Mixed generation working
- ✅ Advanced CLI features implemented
- ✅ Memory management optimized
- ✅ Database diagnostics complete

---

## Risk Mitigation

### Technical Risks
1. **External API Dependencies**: Implement fallback mechanisms
2. **Performance Issues**: Add caching and optimization
3. **Memory Usage**: Implement resource monitoring

### Integration Risks
1. **Database Schema**: Ensure compatibility with existing data
2. **CLI Interface**: Maintain backward compatibility
3. **Configuration**: Provide sensible defaults

---

## Post-Migration Validation

### Feature Parity Check
- [ ] All original playlist generation methods working
- [ ] Performance comparable to original
- [ ] CLI interface fully functional
- [ ] Database operations working correctly

### Quality Assurance
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Memory usage optimized

### Documentation
- [ ] API documentation updated
- [ ] CLI help updated
- [ ] Configuration guide updated
- [ ] Migration notes documented 