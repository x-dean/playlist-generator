import React, { useState } from 'react';
import { 
  Container, 
  Title, 
  Stack,
  Card,
  Table,
  Text,
  Group,
  Badge,
  Button,
  TextInput,
  Select,
  Pagination,
  Loader,
  Alert,
  ActionIcon
} from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { 
  IconSearch, 
  IconMusic, 
  IconClock,
  IconUser,
  IconDisc,
  IconRefresh,
  IconAnalyze
} from '@tabler/icons-react';
import { apiClient } from '../api/client';

interface Track {
  id: string;
  filename: string;
  title?: string;
  artist?: string;
  album?: string;
  duration: number;
  file_size: number;
  audio_features?: Record<string, any>;
  created_at: string;
}

interface TracksResponse {
  tracks: Track[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export const Library = () => {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Fetch tracks
  const { data: tracksData, isLoading, error, refetch } = useQuery<TracksResponse>({
    queryKey: ['tracks', page, search, sortBy, sortOrder],
    queryFn: () => apiClient.get('/api/library/tracks', {
      params: {
        page,
        per_page: 20,
        search: search || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      }
    }).then(res => res.data),
  });

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getFeatureValue = (track: Track, feature: string) => {
    if (!track.audio_features) return 'N/A';
    const value = track.audio_features[feature];
    if (value === undefined || value === null) return 'N/A';
    if (typeof value === 'number') {
      return value.toFixed(1);
    }
    return String(value);
  };

  const analyzeTrack = async (trackId: string) => {
    try {
      await apiClient.post(`/api/library/tracks/${trackId}/analyze`);
      // Refetch to show updated data
      refetch();
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between" align="center">
          <div>
            <Title order={1} c="white">
              📚 Music Library
            </Title>
            <Text c="dimmed" size="lg">
              Browse and manage your music collection
            </Text>
          </div>
          <Button 
            leftSection={<IconRefresh size={16} />}
            onClick={() => refetch()}
            loading={isLoading}
          >
            Refresh
          </Button>
        </Group>

        {/* Filters */}
        <Card withBorder padding="lg">
          <Group gap="md" align="end">
            <TextInput
              placeholder="Search tracks, artists, albums..."
              leftSection={<IconSearch size={16} />}
              value={search}
              onChange={(event) => setSearch(event.currentTarget.value)}
              style={{ flex: 1 }}
            />
            <Select
              label="Sort by"
              value={sortBy}
              onChange={(value) => setSortBy(value || 'created_at')}
              data={[
                { value: 'created_at', label: 'Date Added' },
                { value: 'title', label: 'Title' },
                { value: 'artist', label: 'Artist' },
                { value: 'album', label: 'Album' },
                { value: 'duration', label: 'Duration' },
              ]}
            />
            <Select
              label="Order"
              value={sortOrder}
              onChange={(value) => setSortOrder(value as 'asc' | 'desc' || 'desc')}
              data={[
                { value: 'desc', label: 'Descending' },
                { value: 'asc', label: 'Ascending' },
              ]}
            />
          </Group>
        </Card>

        {/* Content */}
        {error && (
          <Alert color="red" title="Error loading tracks">
            Failed to load music library. Please check if the backend is running.
          </Alert>
        )}

        {isLoading ? (
          <Card withBorder padding="lg">
            <Group justify="center">
              <Loader />
              <Text>Loading music library...</Text>
            </Group>
          </Card>
        ) : tracksData && tracksData.tracks.length > 0 ? (
          <>
            {/* Stats */}
            <Group gap="xl">
              <Text>
                <strong>Total:</strong> {tracksData.total} tracks
              </Text>
              <Text>
                <strong>Page:</strong> {tracksData.page} of {tracksData.total_pages}
              </Text>
            </Group>

            {/* Tracks Table */}
            <Card withBorder padding={0}>
              <Table highlightOnHover verticalSpacing="md">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Track</Table.Th>
                    <Table.Th>Artist</Table.Th>
                    <Table.Th>Album</Table.Th>
                    <Table.Th>Duration</Table.Th>
                    <Table.Th>Size</Table.Th>
                    <Table.Th>Tempo</Table.Th>
                    <Table.Th>Key</Table.Th>
                    <Table.Th>Status</Table.Th>
                    <Table.Th>Actions</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {tracksData.tracks.map((track) => (
                    <Table.Tr key={track.id}>
                      <Table.Td>
                        <Group gap="sm">
                          <IconMusic size={16} color="#4DABF7" />
                          <div>
                            <Text fw={500}>
                              {track.title || track.filename}
                            </Text>
                            {track.title && (
                              <Text size="xs" c="dimmed">
                                {track.filename}
                              </Text>
                            )}
                          </div>
                        </Group>
                      </Table.Td>
                      <Table.Td>
                        <Group gap="xs">
                          <IconUser size={14} />
                          <Text>{track.artist || 'Unknown'}</Text>
                        </Group>
                      </Table.Td>
                      <Table.Td>
                        <Group gap="xs">
                          <IconDisc size={14} />
                          <Text>{track.album || 'Unknown'}</Text>
                        </Group>
                      </Table.Td>
                      <Table.Td>
                        <Group gap="xs">
                          <IconClock size={14} />
                          <Text>{formatDuration(track.duration)}</Text>
                        </Group>
                      </Table.Td>
                      <Table.Td>
                        <Text size="sm">{formatFileSize(track.file_size)}</Text>
                      </Table.Td>
                      <Table.Td>
                        <Text size="sm">{getFeatureValue(track, 'tempo')} BPM</Text>
                      </Table.Td>
                      <Table.Td>
                        <Text size="sm">{getFeatureValue(track, 'key')}</Text>
                      </Table.Td>
                      <Table.Td>
                        <Badge 
                          color={track.audio_features ? 'green' : 'gray'}
                          variant="filled"
                        >
                          {track.audio_features ? 'Analyzed' : 'Pending'}
                        </Badge>
                      </Table.Td>
                      <Table.Td>
                        {!track.audio_features && (
                          <ActionIcon 
                            variant="filled" 
                            color="blue"
                            onClick={() => analyzeTrack(track.id)}
                            title="Analyze track"
                          >
                            <IconAnalyze size={16} />
                          </ActionIcon>
                        )}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Card>

            {/* Pagination */}
            {tracksData.total_pages > 1 && (
              <Group justify="center">
                <Pagination
                  value={page}
                  onChange={setPage}
                  total={tracksData.total_pages}
                />
              </Group>
            )}
          </>
        ) : (
          <Card withBorder padding="xl">
            <Stack align="center" gap="lg">
              <IconMusic size={48} color="#868E96" />
              <div style={{ textAlign: 'center' }}>
                <Title order={3} c="dimmed">No tracks found</Title>
                <Text c="dimmed">
                  {search ? 'No tracks match your search criteria.' : 'Your music library is empty.'}
                </Text>
              </div>
              {!search && (
                <Button variant="outline" onClick={() => refetch()}>
                  Scan for music files
                </Button>
              )}
            </Stack>
          </Card>
        )}
      </Stack>
    </Container>
  );
};