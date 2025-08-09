import React from 'react';
import { 
  Container, 
  Title, 
  Grid, 
  Card, 
  Text, 
  Group, 
  Stack,
  Badge,
  Button,
  Alert,
  Loader
} from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { 
  IconMusic, 
  IconPlaylist, 
  IconChartLine, 
  IconDatabase,
  IconInfoCircle
} from '@tabler/icons-react';
import { apiClient } from '../api/client';

interface SystemHealth {
  status: string;
  timestamp: number;
  version: string;
}

interface LibraryStats {
  total_tracks: number;
  analyzed_tracks: number;
  total_playlists: number;
  total_duration: number;
}

export const Dashboard = () => {
  // Fetch system health
  const { data: health, isLoading: healthLoading } = useQuery<SystemHealth>({
    queryKey: ['health'],
    queryFn: () => apiClient.get('/api/health').then(res => res.data),
  });

  // Fetch library stats
  const { data: stats, isLoading: statsLoading } = useQuery<LibraryStats>({
    queryKey: ['library-stats'],
    queryFn: () => apiClient.get('/api/library/stats').then(res => res.data),
    retry: false,
  });

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <Container size="lg" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between" align="center">
          <div>
            <Title order={1} c="white">
              🎵 Playlista v2 Dashboard
            </Title>
            <Text c="dimmed" size="lg">
              High-Performance Music Analysis & Playlist Generation
            </Text>
          </div>
          {health && (
            <Badge 
              color={health.status === 'healthy' ? 'green' : 'red'} 
              variant="filled"
              size="lg"
            >
              {health.status === 'healthy' ? '✅ System Healthy' : '❌ System Issue'}
            </Badge>
          )}
        </Group>

        {/* System Status */}
        <Card withBorder padding="lg">
          <Group justify="space-between" mb="md">
            <Title order={3}>System Status</Title>
            {healthLoading && <Loader size="sm" />}
          </Group>
          
          {health ? (
            <Group gap="xl">
              <Text>
                <strong>Status:</strong> {health.status}
              </Text>
              <Text>
                <strong>Version:</strong> {health.version}
              </Text>
              <Text>
                <strong>Uptime:</strong> {new Date(health.timestamp * 1000).toLocaleString()}
              </Text>
            </Group>
          ) : (
            <Alert icon={<IconInfoCircle size={16} />} color="blue">
              Connecting to backend...
            </Alert>
          )}
        </Card>

        {/* Library Statistics */}
        <Grid>
          <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
            <Card withBorder padding="lg" h="100%">
              <Group gap="sm" mb="xs">
                <IconMusic size={20} color="#4DABF7" />
                <Text fw={500}>Total Tracks</Text>
              </Group>
              {statsLoading ? (
                <Loader size="sm" />
              ) : (
                <Text size="xl" fw={700} c="blue">
                  {stats?.total_tracks || 0}
                </Text>
              )}
            </Card>
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
            <Card withBorder padding="lg" h="100%">
              <Group gap="sm" mb="xs">
                <IconChartLine size={20} color="#51CF66" />
                <Text fw={500}>Analyzed</Text>
              </Group>
              {statsLoading ? (
                <Loader size="sm" />
              ) : (
                <Text size="xl" fw={700} c="green">
                  {stats?.analyzed_tracks || 0}
                </Text>
              )}
            </Card>
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
            <Card withBorder padding="lg" h="100%">
              <Group gap="sm" mb="xs">
                <IconPlaylist size={20} color="#FFD43B" />
                <Text fw={500}>Playlists</Text>
              </Group>
              {statsLoading ? (
                <Loader size="sm" />
              ) : (
                <Text size="xl" fw={700} c="yellow">
                  {stats?.total_playlists || 0}
                </Text>
              )}
            </Card>
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
            <Card withBorder padding="lg" h="100%">
              <Group gap="sm" mb="xs">
                <IconDatabase size={20} color="#FF6B6B" />
                <Text fw={500}>Duration</Text>
              </Group>
              {statsLoading ? (
                <Loader size="sm" />
              ) : (
                <Text size="xl" fw={700} c="red">
                  {stats?.total_duration ? formatDuration(stats.total_duration) : '0h 0m'}
                </Text>
              )}
            </Card>
          </Grid.Col>
        </Grid>

        {/* Quick Actions */}
        <Card withBorder padding="lg">
          <Title order={3} mb="md">Quick Actions</Title>
          <Group gap="md">
            <Button 
              variant="filled" 
              color="blue" 
              leftSection={<IconMusic size={16} />}
              component="a"
              href="/library"
            >
              Browse Library
            </Button>
            <Button 
              variant="filled" 
              color="green" 
              leftSection={<IconChartLine size={16} />}
              component="a"
              href="/analysis"
            >
              Start Analysis
            </Button>
            <Button 
              variant="filled" 
              color="yellow" 
              leftSection={<IconPlaylist size={16} />}
              component="a"
              href="/playlists"
            >
              Generate Playlist
            </Button>
          </Group>
        </Card>

        {/* Feature Overview */}
        <Card withBorder padding="lg">
          <Title order={3} mb="md">Features</Title>
          <Grid>
            <Grid.Col span={{ base: 12, md: 4 }}>
              <Text fw={500} mb="xs">🔬 Advanced Analysis</Text>
              <Text size="sm" c="dimmed">
                27+ audio features including tempo, key, mood, and spectral analysis
              </Text>
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 4 }}>
              <Text fw={500} mb="xs">🤖 ML-Powered</Text>
              <Text size="sm" c="dimmed">
                Genre classification, mood detection, and audio embeddings
              </Text>
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 4 }}>
              <Text fw={500} mb="xs">🎵 Smart Playlists</Text>
              <Text size="sm" c="dimmed">
                Multiple algorithms for similarity, clustering, and mood-based generation
              </Text>
            </Grid.Col>
          </Grid>
        </Card>
      </Stack>
    </Container>
  );
};