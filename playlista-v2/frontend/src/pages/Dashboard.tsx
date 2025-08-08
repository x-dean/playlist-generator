import React from 'react';
import { Container, Title, Grid, Card, Text, Badge, Group } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';

export function Dashboard() {
  const { data: libraryStats } = useQuery({
    queryKey: ['library-stats'],
    queryFn: () => api.get('/library/stats').then(res => res.data)
  });

  const { data: analysisStatus } = useQuery({
    queryKey: ['analysis-status'],
    queryFn: () => api.get('/analysis/status').then(res => res.data)
  });

  return (
    <Container size="xl">
      <Title order={1} mb="xl">Dashboard</Title>
      
      <Grid>
        <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
          <Card shadow="sm" padding="lg" radius="md" withBorder>
            <Text size="sm" color="dimmed">Total Tracks</Text>
            <Text size="xl" fw={700}>{libraryStats?.total_tracks || 0}</Text>
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
          <Card shadow="sm" padding="lg" radius="md" withBorder>
            <Text size="sm" color="dimmed">Analyzed</Text>
            <Group>
              <Text size="xl" fw={700}>{libraryStats?.analyzed_tracks || 0}</Text>
              <Badge color="blue">
                {libraryStats?.analysis_percentage || 0}%
              </Badge>
            </Group>
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
          <Card shadow="sm" padding="lg" radius="md" withBorder>
            <Text size="sm" color="dimmed">Average BPM</Text>
            <Text size="xl" fw={700}>
              {libraryStats?.average_features?.bpm?.toFixed(0) || 'N/A'}
            </Text>
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
          <Card shadow="sm" padding="lg" radius="md" withBorder>
            <Text size="sm" color="dimmed">Engine Status</Text>
            <Badge color={analysisStatus?.engine_status === 'running' ? 'green' : 'red'}>
              {analysisStatus?.engine_status || 'Unknown'}
            </Badge>
          </Card>
        </Grid.Col>
      </Grid>
    </Container>
  );
}
