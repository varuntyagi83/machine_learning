-- dbt SQL Model
-- Example: Transform and create data model
WITH shipments_cte AS (
    SELECT
        s.ShipmentID,
        s.CustomerID,
        s.ProductID,
        s.TransportationMode,
        s.ShipmentDate,
        b.BookingDate,
        sa.AgentName
    FROM
        STAGING_TABLE_SHIPMENTS s
    LEFT JOIN
        STAGING_TABLE_BOOKINGS b ON s.ShipmentID = b.ShipmentID
    LEFT JOIN
        STAGING_TABLE_SALES_AGENTS sa ON b.AgentID = sa.AgentID
)

-- Additional dbt transformations and logic as needed

SELECT
    ShipmentID,
    CustomerID,
    ProductID,
    TransportationMode,
    ShipmentDate,
    BookingDate,
    AgentName
FROM
    shipments_cte
